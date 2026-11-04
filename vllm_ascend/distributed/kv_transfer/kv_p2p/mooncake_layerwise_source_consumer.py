# SPDX-License-Identifier: Apache-2.0
"""P-side source consumer adapting the reuse source contract to the
layerwise connector's existing slot ledger and send path.

The Store-side ``ReuseSourceRegistry`` drives this consumer through the five
source-lifecycle callbacks. The adapter maps them onto the connector worker's
already-established slot wait / send-done machinery so that enabling the source
contract does not start a second, parallel send path.

Send deduplication: when the contract is active, the layer's transfer is
published through ``source_ready`` (readers registered before the async task is
released). The legacy ``save_kv_layer`` hook still owns the actual DMA
submission; a per-source flag records that the source generation has been
published so the send path does not double-publish. ``wait_source`` delegates
to the worker's ``wait_for_layer_reuse``, which already enforces the slot
ledger's three-branch rule and bounded wait.
"""

import threading
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
    LayerSource,
    RequestSource,
)


class LayerwiseSourceConsumer:
    """Adapts the five-method source contract to the layerwise send path.

    The consumer is installed on the producer connector and called by the
    Store registry. Callbacks may run on receiver or model threads, so the
    per-generation state is guarded by a lock. The actual DMA submission stays
    in the existing ``save_kv_layer`` hook; this consumer only tracks source
    generation lifecycle and quiescence.
    """

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._lock = threading.RLock()
        # epochs -> (sealed, pending_readers, ready_event, error)
        self._generations: dict[tuple, dict] = {}
        # layer -> active epochs, so wait_source resolves a layer to its source
        self._layer_source: dict[int, tuple] = {}

    def _key(self, source: LayerSource) -> tuple:
        return source.epochs

    def open_source(self, source: LayerSource) -> None:
        """Nonblocking registration of a new source generation before first write."""
        with self._lock:
            self._generations[self._key(source)] = {
                "sealed": False,
                "pending": 0,
                "ready_event": None,
                "error": None,
            }
            self._layer_source[source.layer] = self._key(source)

    def source_ready(self, source: LayerSource, requests: tuple[RequestSource, ...], ready_event: Any) -> None:
        """Register all batch readers before the async send task is released.

        The ready_event is the KV-ready fence recorded by the Store. The actual
        DMA submission remains in the legacy save hook, which checks the
        published flag to avoid double-sending this generation.
        """
        with self._lock:
            record = self._generations.get(self._key(source))
            if record is None:
                raise RuntimeError("source_ready before open_source for the layer generation")
            record["pending"] = len(requests)
            record["ready_event"] = ready_event
            record["published"] = True

    def seal_source(self, source: LayerSource) -> None:
        """Close additions for this generation, including an empty batch."""
        with self._lock:
            record = self._generations.get(self._key(source))
            if record is None:
                raise RuntimeError("seal_source before open_source for the layer generation")
            record["sealed"] = True

    def wait_source(self, source: LayerSource) -> None:
        """Succeed only after sealing AND all readers quiesce successfully.

        Delegates to the worker's existing layer reuse wait, which enforces the
        slot ledger's three-branch rule (current/older/pending) and a bounded
        timeout. A failed generation surfaces its error through that path.
        """
        with self._lock:
            record = self._generations.get(self._key(source))
            error = record["error"] if record is not None else None
        if error is not None:
            raise RuntimeError("Reuse source generation failed; allocations retained") from error
        # The worker's wait already resolves the layer to its slots and applies
        # the three-branch rule; it raises on failure or timeout.
        self._worker.wait_for_layer_reuse(source.layer)

    def abort_source(self, source: LayerSource, error: Exception) -> None:
        """Mark the generation failed while retaining in-flight references."""
        with self._lock:
            record = self._generations.get(self._key(source))
            if record is not None:
                record["error"] = error
        # Complete the worker's reuse state as failed so waiters wake and the
        # slots are not silently reused. The worker keeps in-flight references
        # until the failure is observed.
        complete = getattr(self._worker, "_complete_layer_reuse", None)
        if callable(complete):
            complete(source.layer, str(error))

    def is_published(self, layer: int) -> bool:
        """True when source_ready has fired for the layer's current generation.

        The legacy save hook consults this to avoid double-submitting a
        generation already published through the source contract.
        """
        with self._lock:
            key = self._layer_source.get(layer)
            if key is None:
                return False
            record = self._generations.get(key)
            return bool(record and record.get("published"))

    def retire(self, layer: int) -> None:
        """Drop the generation record after the layer's reuse is complete."""
        with self._lock:
            key = self._layer_source.pop(layer, None)
            if key is not None:
                self._generations.pop(key, None)
