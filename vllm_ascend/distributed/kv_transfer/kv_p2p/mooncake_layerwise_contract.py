# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Layerwise PD transfer contract shared between the connector side and the target side.

Owner: the connector side (sole naming & maintenance authority); consumed by
the target side to implement ``TargetProvider``. This file is the interface
authority for the layerwise PD transfer.
- ``get_target_views`` returns a flat ``tuple[CacheView, ...]``;
  consumers group by ``CacheView.component`` themselves.
- ``release_target(lease) -> None``; its normal return is the provider's
  promise that target-side references (physical pool + write-back) are already
  dropped.
Hard rule: pure stdlib. No torch / torch_npu / zmq / vllm / msgspec imports, so
this module (and its unit tests) run in CPU-only CI.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
VISIBILITY_CONTRACT = "stream-dependency"

_MEMORY_KINDS = ("host", "device")


class ContractError(Exception):
    """Contract violation.

    Startup time -> abort startup (no silent degradation).
    Request time -> fail that request only.
    """


def _require_int(name: str, value: object, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int (bool is not accepted), got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{name} must be <= {maximum}, got {value}")
    return value


def _require_str(name: str, value: object, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ContractError(f"{name} must not be empty")
    return value


def _require_opt_str(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_str(name, value)


def _normalize_seq(name: str, value: object) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise ContractError(f"{name} must be a tuple or list, got {type(value).__name__}")


class Component(str, Enum):
    """KV component carried by a transfer. Never infer from group_id: with a
    single group, dsa_main_group_id == dsa_indexer_group_id == 0."""

    MAIN = "main"
    INDEXER = "indexer"

    def to_phase(self) -> str:
        """Map to the repo's ``DsaTransferPhase`` value (mooncake_dsa_metadata.py).

        Returned as plain str so this module stays vllm-free; str-Enum comparison
        makes it equal to ``DsaTransferPhase.MAIN_D2RH`` / ``.INDEXER_D2D``.
        """
        return "main_d2rh" if self is Component.MAIN else "indexer_d2d"


class Status(str, Enum):
    """Terminal transfer states. Once written into a ledger, never re-judged."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TokenRange:
    """Half-open token span [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _require_int("TokenRange.start", self.start, minimum=0)
        _require_int("TokenRange.end", self.end, minimum=0)
        if self.start > self.end:
            raise ContractError(f"TokenRange start({self.start}) > end({self.end})")


@dataclass(frozen=True)
class RequestKey:
    """Request identity. ``request_instance`` is generated once per request
    lifetime on the D scheduler (uuid4().hex) and reaches P via
    kv_transfer_params; ``transfer_generation`` is allocated by D when the
    target is published (initial 0), bumped only on preempt/retry."""

    request_instance: str
    transfer_generation: int

    def __post_init__(self) -> None:
        _require_str("RequestKey.request_instance", self.request_instance)
        _require_int("RequestKey.transfer_generation", self.transfer_generation, minimum=0)


@dataclass(frozen=True)
class TargetScope:
    """Where a transfer lands. MAIN uses ``consumer_rank=None``; INDEXER fills
    the consuming rank. (backing_id, backing_generation) must be part of the
    transfer identity or a second result on the same layer overwrites the first."""

    backing_id: str
    backing_generation: int
    consumer_rank: int | None = None

    def __post_init__(self) -> None:
        _require_str("TargetScope.backing_id", self.backing_id)
        _require_int("TargetScope.backing_generation", self.backing_generation, minimum=0)
        if self.consumer_rank is not None:
            _require_int("TargetScope.consumer_rank", self.consumer_rank, minimum=0)


@dataclass(frozen=True)
class TransferKey:
    """Seven-field identity of one (layer, component, range) transfer.
    Frozen and hashable; all seven fields participate."""

    request: RequestKey
    rank: int
    logical_layer: int
    operation_id: int
    component: Component
    token_range: TokenRange
    target_scope: TargetScope

    def __post_init__(self) -> None:
        _require_int("TransferKey.rank", self.rank, minimum=0)
        _require_int("TransferKey.logical_layer", self.logical_layer, minimum=0)
        _require_int("TransferKey.operation_id", self.operation_id, minimum=0)
        if not isinstance(self.component, Component):
            raise ContractError(f"TransferKey.component must be Component, got {type(self.component).__name__}")
        if not isinstance(self.token_range, TokenRange):
            raise ContractError(f"TransferKey.token_range must be TokenRange, got {type(self.token_range).__name__}")
        if not isinstance(self.target_scope, TargetScope):
            raise ContractError(f"TransferKey.target_scope must be TargetScope, got {type(self.target_scope).__name__}")
        if self.component is Component.MAIN and self.target_scope.consumer_rank is not None:
            raise ContractError("MAIN transfers must use TargetScope.consumer_rank=None")
        if self.component is Component.INDEXER and self.target_scope.consumer_rank is None:
            raise ContractError("INDEXER transfers must fill TargetScope.consumer_rank")


@dataclass(frozen=True)
class BlockAllocation:
    """D-scheduler allocation facts handed to the provider's ``acquire_target``.
    The connector is read-only here. ``group_block_ids`` is group-major."""

    request: RequestKey
    allocation_generation: int
    group_block_ids: tuple[tuple[int, ...], ...]
    manager_block_size_tokens: int
    prompt_tokens: int
    retention_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestKey):
            raise ContractError(f"BlockAllocation.request must be RequestKey, got {type(self.request).__name__}")
        _require_int("BlockAllocation.allocation_generation", self.allocation_generation, minimum=0)
        groups = _normalize_seq("BlockAllocation.group_block_ids", self.group_block_ids)
        normalized_groups = tuple(_normalize_seq(f"group[{i}]", g) for i, g in enumerate(groups))
        object.__setattr__(self, "group_block_ids", normalized_groups)
        _require_int("BlockAllocation.manager_block_size_tokens", self.manager_block_size_tokens, minimum=1)
        _require_int("BlockAllocation.prompt_tokens", self.prompt_tokens, minimum=1)
        retention_token = _require_opt_str("BlockAllocation.retention_token", self.retention_token)
        object.__setattr__(self, "retention_token", retention_token)


@dataclass
class LocalRegion:
    """Process-local registration result. Never serialized."""

    allocation_id: int
    allocation_generation: int
    base_address: int
    length_bytes: int
    owner_lease: Any = None

    def __post_init__(self) -> None:
        _require_int("LocalRegion.allocation_id", self.allocation_id, minimum=0)
        _require_int("LocalRegion.allocation_generation", self.allocation_generation, minimum=0)
        _require_int("LocalRegion.base_address", self.base_address, minimum=0)
        _require_int("LocalRegion.length_bytes", self.length_bytes, minimum=1)


@dataclass(frozen=True)
class RegisteredRegion:
    """One registered memory region published to the remote side."""

    engine_instance: str
    registration_generation: int
    region_id: str
    backing_id: str
    backing_generation: int
    backing_offset_bytes: int
    length_bytes: int
    memory_kind: str

    def __post_init__(self) -> None:
        _require_str("RegisteredRegion.engine_instance", self.engine_instance)
        _require_int("RegisteredRegion.registration_generation", self.registration_generation, minimum=0)
        _require_str("RegisteredRegion.region_id", self.region_id)
        _require_str("RegisteredRegion.backing_id", self.backing_id)
        _require_int("RegisteredRegion.backing_generation", self.backing_generation, minimum=0)
        _require_int("RegisteredRegion.backing_offset_bytes", self.backing_offset_bytes, minimum=0)
        _require_int("RegisteredRegion.length_bytes", self.length_bytes, minimum=1)
        if self.memory_kind not in _MEMORY_KINDS:
            raise ContractError(
                f"RegisteredRegion.memory_kind must be one of {_MEMORY_KINDS}, got {self.memory_kind!r}"
            )


@dataclass(frozen=True)
class CacheView:
    """How a component's layer KV lays out inside a registered region."""

    component: Component
    logical_layer: int
    region_binding: str
    dtype: str
    strides_bytes: tuple[int, ...]
    page_size_tokens: int
    manager_block_size_tokens: int
    block_capacity: int
    region_offset_bytes: int
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.component, Component):
            raise ContractError(f"CacheView.component must be Component, got {type(self.component).__name__}")
        _require_int("CacheView.logical_layer", self.logical_layer, minimum=0)
        _require_str("CacheView.region_binding", self.region_binding)
        _require_str("CacheView.dtype", self.dtype)
        object.__setattr__(self, "strides_bytes", _normalize_seq("CacheView.strides_bytes", self.strides_bytes))
        for i, s in enumerate(self.strides_bytes):
            _require_int(f"CacheView.strides_bytes[{i}]", s, minimum=0)
        _require_int("CacheView.page_size_tokens", self.page_size_tokens, minimum=1)
        _require_int("CacheView.manager_block_size_tokens", self.manager_block_size_tokens, minimum=1)
        if self.manager_block_size_tokens % self.page_size_tokens != 0:
            # First-version restriction; startup validation also fails fast on this.
            raise ContractError(
                f"manager_block_size_tokens({self.manager_block_size_tokens}) must be a multiple of "
                f"page_size_tokens({self.page_size_tokens})"
            )
        _require_int("CacheView.block_capacity", self.block_capacity, minimum=0)
        _require_int("CacheView.region_offset_bytes", self.region_offset_bytes, minimum=0)
        object.__setattr__(self, "block_ids", _normalize_seq("CacheView.block_ids", self.block_ids))


@dataclass(frozen=True)
class WriterPlan:
    """Who writes which backing over which ranges. Ranks
    ``writer_ranks[destination_ordinal % len(writer_ranks)]``; the connector builds this and
    proves coverage at startup. Treat inner dicts as immutable."""

    writers_by_backing: dict[str, tuple[int, ...]] = field(default_factory=dict)
    ranges_by_backing: dict[str, tuple[TokenRange, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetLease:
    """Acquired target: binds RegisteredRegions + CacheViews + WriterPlan to one
    request. Regions must not repeat (backing_id, backing_generation)."""

    contract_version: int
    request: RequestKey
    rank: int
    allocation: BlockAllocation
    regions: tuple[RegisteredRegion, ...]
    views: tuple[CacheView, ...]
    writer_plan: WriterPlan

    def __post_init__(self) -> None:
        _require_int("TargetLease.contract_version", self.contract_version, minimum=1)
        if not isinstance(self.request, RequestKey):
            raise ContractError(f"TargetLease.request must be RequestKey, got {type(self.request).__name__}")
        _require_int("TargetLease.rank", self.rank, minimum=0)
        if not isinstance(self.allocation, BlockAllocation):
            raise ContractError(f"TargetLease.allocation must be BlockAllocation, got {type(self.allocation).__name__}")
        object.__setattr__(self, "regions", _normalize_seq("TargetLease.regions", self.regions))
        object.__setattr__(self, "views", _normalize_seq("TargetLease.views", self.views))
        for i, r in enumerate(self.regions):
            if not isinstance(r, RegisteredRegion):
                raise ContractError(f"TargetLease.regions[{i}] must be RegisteredRegion, got {type(r).__name__}")
        seen: set[tuple[str, int]] = set()
        for r in self.regions:
            pair = (r.backing_id, r.backing_generation)
            if pair in seen:
                raise ContractError(f"TargetLease.regions repeats (backing_id, backing_generation)={pair}")
            seen.add(pair)
        for i, v in enumerate(self.views):
            if not isinstance(v, CacheView):
                raise ContractError(f"TargetLease.views[{i}] must be CacheView, got {type(v).__name__}")
        if not isinstance(self.writer_plan, WriterPlan):
            raise ContractError(f"TargetLease.writer_plan must be WriterPlan, got {type(self.writer_plan).__name__}")


@dataclass
class ReceiveReceipt:
    """D-side readiness receipt handed to the provider's ``prepare_decode``.
    ``visible`` starts False and is set True only after prepare_decode returns
    normally (normal return = consumer stream dependency wired in)."""

    request: RequestKey
    target_generation: int
    main_union_ok: bool
    indexer_ranks_ok: bool
    terminals_ok: bool
    visible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestKey):
            raise ContractError(f"ReceiveReceipt.request must be RequestKey, got {type(self.request).__name__}")
        _require_int("ReceiveReceipt.target_generation", self.target_generation, minimum=0)
        for name in ("main_union_ok", "indexer_ranks_ok", "terminals_ok", "visible"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"ReceiveReceipt.{name} must be bool")


@dataclass(frozen=True)
class ReleaseAck:
    """Safe-release confirmation returned to the scheduler .
    Produced by the connector: DMA quiescence from its ledger, lease
    refs per the provider's release_target promise, retention from the scheduler."""

    safe_to_release: bool
    retention_token: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.safe_to_release, bool):
            raise ContractError("ReleaseAck.safe_to_release must be bool")
        retention_token = _require_opt_str("ReleaseAck.retention_token", self.retention_token)
        object.__setattr__(self, "retention_token", retention_token)
        _require_str("ReleaseAck.reason", self.reason, allow_empty=True)


@dataclass(frozen=True)
class TransferResult:
    """One (layer, component, range) transfer outcome.
    Carries the full TransferKey so the D-side ledger can judge by identity."""

    key: TransferKey
    status: Status
    source_quiesced: bool = False
    error_code: int | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, TransferKey):
            raise ContractError(f"TransferResult.key must be TransferKey, got {type(self.key).__name__}")
        if not isinstance(self.status, Status):
            raise ContractError(f"TransferResult.status must be Status, got {type(self.status).__name__}")
        if not isinstance(self.source_quiesced, bool):
            raise ContractError("TransferResult.source_quiesced must be bool")
        if self.error_code is not None:
            _require_int("TransferResult.error_code", self.error_code)
        object.__setattr__(self, "error_message", _require_opt_str("TransferResult.error_message", self.error_message))


@runtime_checkable
class TargetProvider(Protocol):
    """Provider interface for the D-side target runtime.

    Implemented on the D-side connector worker; the connector discovers the object,
    checks capabilities at startup, and calls it per request.
    """

    contract_version: int
    visibility_contract: str
    supports_layerwise_buffer_reuse: bool

    def acquire_target(self, request: RequestKey, allocation: BlockAllocation) -> TargetLease:
        """Acquire the D-side target for one request. Must cover the full prompt
        for both components or raise; no partial targets."""
        ...

    def get_target_views(self, lease: TargetLease) -> tuple[CacheView, ...]:
        """Return the CacheViews as a flat tuple. Each view carries its own
        ``region_binding`` and ``component``; never rely on list order."""
        ...

    def prepare_decode(self, lease: TargetLease, receipt: ReceiveReceipt) -> None:
        """Validate identity and wire the consumer stream dependency.
        Normal return = dependency is wired. Raise on anything unsafe; raising
        must leave the request invisible (never published)."""
        ...

    def release_target(self, lease: TargetLease) -> None:
        """Initiate release. Normal return is the provider's promise that target-side
        references (physical pool refs + write-back readers) are already
        dropped; the connector derives the scheduler-facing ReleaseAck from this promise
        plus its own DMA quiescence. Raise if unsafe to release."""
        ...


_PROVIDER_METHODS = ("acquire_target", "get_target_views", "prepare_decode", "release_target")


def assert_provider_capabilities(provider: Any) -> None:
    """Startup gate four methods callable + version/visibility/reuse.
    Raises ContractError listing every missing capability -> startup aborts."""

    problems: list[str] = []
    for method in _PROVIDER_METHODS:
        if not callable(getattr(provider, method, None)):
            problems.append(f"missing/uncallable method: {method}()")
    version = getattr(provider, "contract_version", None)
    if isinstance(version, bool) or not isinstance(version, int):
        problems.append(f"contract_version must be int, got {type(version).__name__}")
    elif version != CONTRACT_VERSION:
        problems.append(f"contract_version mismatch: provider={version}, required={CONTRACT_VERSION}")
    visibility = getattr(provider, "visibility_contract", None)
    if visibility != VISIBILITY_CONTRACT:
        problems.append(f"visibility_contract mismatch: provider={visibility!r}, required={VISIBILITY_CONTRACT!r}")
    if getattr(provider, "supports_layerwise_buffer_reuse", None) is not True:
        problems.append(
            "supports_layerwise_buffer_reuse must be True, got "
            f"{getattr(provider, 'supports_layerwise_buffer_reuse', None)!r}"
        )
    if problems:
        raise ContractError("TargetProvider capability check failed:\n  - " + "\n  - ".join(problems))


def assert_lease_consistent(lease: TargetLease, request: RequestKey) -> None:
    """Per-request gate before the target is used. Raises
    ContractError -> that request fails; other requests continue."""

    if isinstance(lease.contract_version, bool) or not isinstance(lease.contract_version, int):
        raise ContractError(f"lease.contract_version must be int, got {type(lease.contract_version).__name__}")
    if lease.contract_version != CONTRACT_VERSION:
        raise ContractError(f"lease.contract_version mismatch: {lease.contract_version} != {CONTRACT_VERSION}")
    if lease.request != request:
        raise ContractError(f"lease belongs to {lease.request}, expected {request}")
    seen: set[tuple[str, int]] = set()
    for r in lease.regions:
        pair = (r.backing_id, r.backing_generation)
        if pair in seen:
            raise ContractError(f"lease regions repeat (backing_id, backing_generation)={pair}")
        seen.add(pair)
    region_ids = {r.region_id for r in lease.regions}
    for view in lease.views:
        if view.region_binding not in region_ids:
            raise ContractError(
                f"view (component={view.component.value}, layer={view.logical_layer}) binds unknown region "
                f"{view.region_binding!r}; known regions={sorted(region_ids)}"
            )


__all__ = [
    "CONTRACT_VERSION",
    "VISIBILITY_CONTRACT",
    "BlockAllocation",
    "CacheView",
    "Component",
    "ContractError",
    "LocalRegion",
    "ReceiveReceipt",
    "RegisteredRegion",
    "ReleaseAck",
    "RequestKey",
    "Status",
    "TargetLease",
    "TargetProvider",
    "TargetScope",
    "TokenRange",
    "TransferKey",
    "TransferResult",
    "WriterPlan",
    "assert_lease_consistent",
    "assert_provider_capabilities",
]
