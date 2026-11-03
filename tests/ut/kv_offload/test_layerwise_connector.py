# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Connector-level tests for the layerwise receipt protocol.

These need the real connector module (torch / torch_npu / vllm / mooncake), so
they skip on CPU-only boxes — the ledger/contract logic they rely on is covered
by test_layerwise_receipt.py, which runs anywhere. The no-compute marking
case exercises save_kv_layer end to end and belongs on the NPU CI.
"""

import pytest

npu = pytest.importorskip("torch_npu")
msgspec = pytest.importorskip("msgspec")

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as lc  # noqa: E402


def _encode_decode(msg):
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=tuple)
    return decoder.decode(encoder.encode(msg))


class WireFormatTests:
    """Positional arrays only; the decoder is Decoder(type=tuple),
    so a map-encoded message must fail loudly, never silently mis-bind."""

    def test_layer_result_roundtrip(self):
        msg = (
            lc.LAYER_RESULT_MSG,
            lc.CONTRACT_VERSION,
            "instance-1",
            0,
            2,
            3,
            3,
            "main",
            0,
            8,
            "b0",
            1,
            "success",
            True,
        )
        assert _encode_decode(msg) == msg

    def test_rank_terminal_roundtrip(self):
        msg = (lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION, "instance-1", 0, 2, "success")
        assert _encode_decode(msg) == msg

    def test_map_encoded_message_is_rejected(self):
        import msgspec as msgspec_mod

        encoder = msgspec_mod.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=tuple)
        with pytest.raises(msgspec.ValidationError):
            decoder.decode(encoder.encode({"type": lc.LAYER_RESULT_MSG, "version": lc.CONTRACT_VERSION}))


class ReceiptHandlerTests:
    """Drive the receive-thread handlers directly (no zmq): malformed frames
    are logged and dropped, version mismatches never reach the ledger."""

    def _thread(self):
        thread = object.__new__(lc.KVCacheRecvingLayerThread)
        import threading

        thread.lock = threading.Lock()
        thread.done_requests = set()
        thread.failed_requests = set()
        thread.task_tracker = {}
        thread.ledger_by_req = {}
        thread.instance_to_req = {}
        thread.receipt_provider = None
        return thread

    def test_version_mismatch_is_dropped(self):
        thread = self._thread()
        msg = (lc.LAYER_RESULT_MSG, 999, "inst", 0, 0, 0, 0, "main", 0, 8, "b0", 1, "success", True)
        thread.handle_layer_result(msg)  # must not raise
        assert not thread.ledger_by_req

    def test_malformed_frame_is_dropped(self):
        thread = self._thread()
        thread.handle_layer_result((lc.LAYER_RESULT_MSG, lc.CONTRACT_VERSION, "inst"))  # too short
        thread.handle_rank_terminal((lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION))  # too short

    def test_unknown_instance_is_dropped(self):
        thread = self._thread()
        msg = (lc.LAYER_RESULT_MSG, lc.CONTRACT_VERSION, "ghost", 0, 0, 0, 0, "main", 0, 8, "b0", 1, "success", True)
        thread.handle_layer_result(msg)
        assert not thread.done_requests


class PrefixHitNoComputeTests:
    """Prefix-hit a layer whose requests are all fully prefix-hit has
    no compute reader — it must be marked no_compute_reader and must not hang
    the send path waiting on a compute-done event that will never come."""

    def test_no_compute_layer_is_marked(self):
        pytest.importorskip("vllm")
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)

        worker.vllm_config = None  # set below via a stub config
        worker.current_layer = 0
        worker.total_layers = 4
        worker.index_to_name = {0: ["layer0"]}
        worker.layer_metadata = {}
        worker._slot_ledger = None
        worker._slot_ledger_active = False
        worker._layer_keys = {}
        worker._compute_done_events = [None] * 4
        worker._no_compute_layers = set()
        worker._cache_write_events = [None] * 4
        worker.kv_send_layer_thread = None
        worker._reqs_need_send_layerwise = {}
        worker.use_mla = False
        worker.kv_cache_specs = []
        worker.block_size = [16]
        worker.sparse_sfa_c8_layers = set()
        worker.enable_kv_quant = False
        worker.enable_c8_quant = False
        worker.pd_head_ratio = 1
        worker.resharding_stream = None

        class _Cfg:
            class kv_transfer_config:  # noqa: N801
                is_kv_producer = True
                is_kv_consumer = False

        class _Meta:
            pass

        req = lc.ReqMeta(
            local_block_ids=[[0, 1]],
            token_ids=[],
            remote_block_ids=[],
            remote_block_size=[],
            remote_engine_id=None,
            remote_host=None,
            remote_port=None,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
            prompt_len=8,
            local_computed_tokens=8,  # fully prefix-hit
        )
        metadata = lc.MooncakeLayerwiseConnectorMetadata()
        metadata.requests["req-1"] = req
        worker.vllm_config = _Cfg()
        worker.save_kv_layer("layer0", [], None, metadata)
        assert 0 in worker._no_compute_layers
        assert worker._compute_done_events[0] is None  # no compute-done event recorded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
