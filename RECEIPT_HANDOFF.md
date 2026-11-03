# receipt 逐层链路交接（connector side → store side）

本文面向负责 store 侧目标管理的开发者，说明 connector 侧已实现的 receipt 链路、待 store 侧实现的接口、接入方式与验收边界。
交接代码版本为 `e26962a`，分支 `feat/layerwise-receipt`（`YYangY/vllm-ascend`），基线 `4234a1f1`（与 `p-main-buffer-reuse` 相同）。
接口准绳为 `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_contract.py`：类型、TargetProvider 协议与校验函数均以该文件为准。

## 1. 交接范围

已实现（connector 侧）：

- 共享契约、双账本、逐层回执协议、provider 注入与启动校验、测试替身与单元测试（§2）。
- receipt 规格的跨进程下发、请求级 release 入口、发布路径缺陷修复（§2）。

待实现（store 侧）：TargetProvider 四方法与三属性（§3）——receipt 全链当前唯一的缺口。

未包含（后续轮次）：connector 侧发送线程对五方法源接口（`open_source` 等）的接入。本链路不经过该接口；store 侧自动安装发现无 version-1 consumer 时静默跳过，无需等待本侧。

## 2. 分支与提交结构

| 提交 | 内容 |
|---|---|
| `a5581b9` | 契约文件；双账本（ReceiptLedger / SlotLedger）；逐层回执协议（LAYER_RESULT / RANK_TERMINAL，receipt 请求 DONE 携带版本号分流）；provider 注入三支判定；FakeTargetProvider；27 项纯 stdlib 单元测试 |
| `e26962a` | receipt 规格经 connector metadata 跨进程下发（worker 侧建账本）；`request_instance` 写入 kv_transfer_params；请求结束调用 `release_target`，失败时保留租约并延迟释放块；修复 `_maybe_publish_target` 中 `backings` 未定义；注释全部自包含；ruff 0.14.0 通过 |
| `f4f534e` | 本文档 |

关键文件：

| 文件 | 职责 |
|---|---|
| `kv_p2p/mooncake_layerwise_contract.py` | 类型、协议、校验（接口准绳） |
| `kv_p2p/mooncake_layerwise_receipt.py` | ReceiptLedger（D 侧，按七字段身份判就绪）；SlotLedger（P 侧，槽位代次栅栏） |
| `kv_p2p/mooncake_layerwise_connector.py` | 回执协议、发布与释放路径、插入点 |
| `kv_p2p/fake_target_provider.py` | TargetProvider 测试替身（实现形态参考） |
| `ascend_multi_connector.py` | `_configure_target_provider` 注入与启动校验 |
| `tests/ut/kv_offload/test_layerwise_receipt.py` | 27 项测试，`python tests/ut/kv_offload/test_layerwise_receipt.py` 本机可复跑 |

## 3. 待 store 侧实现：TargetProvider

### 3.1 属性

```python
contract_version = 1
visibility_contract = "stream-dependency"
supports_layerwise_buffer_reuse = True
```

### 3.2 方法

| 方法 | 语义 |
|---|---|
| `acquire_target(request, allocation) -> TargetLease` | 覆盖全 prompt；容量不足 raise；不返回部分目标 |
| `get_target_views(lease) -> tuple[CacheView, ...]` | 平铺返回；消费方按 `view.component` 分组 |
| `prepare_decode(lease, receipt) -> None` | 校验身份并接入消费 stream 依赖；正常返回即依赖已接；raise 则该请求不发布 |
| `release_target(lease) -> None` | 正常返回 = store 侧引用（物理池 + 写回读者）已同步释放；不安全则 raise |

### 3.3 数据约束

TargetLease 经 connector metadata 随调度跨进程序列化，必须为纯数据（可 pickle）；运行时对象（锁、句柄）不得放入。

### 3.4 安装位置与启动校验

挂在 D 节点的 connector 或 worker 上、暴露 `acquire_target` 即可（扫描范围含两级对象）。注入时执行 `assert_provider_capabilities`，三支判定：

- receipt 模式未开启且无 provider：跳过（普通模式，合法）；
- receipt 模式开启且无 provider：拒绝启动；
- 多于一个 provider：拒绝启动。

## 4. 缺失该实现时的链路行为

- 启动：开启 `layerwise_receipt_mode` 后无 provider 即启动失败（拒绝半套上线）。
- 放行：`visible` 是 `prepare_decode` 的产出而非前提。数据齐备后链路为：data ready → receipt(visible=False) → `prepare_decode` → 正常返回后置 visible → decode 放行。无该调用则请求不会终止。
- 释放：`release_target` 返回后同地址可能被下一请求复用重写；引用未同步清空会产生在途脏读。

## 5. 待确认事项（交付时一并回复）

1. `prepare_decode` 为真实消费 stream 依赖接入，或空壳；
2. `acquire_target` 容量不足的异常类型，及是否存在部分成功路径（本侧按"异常 = 整请求失败"处理）；
3. `release_target` 是否同步返回（返回即引用已清）。

## 6. 集成

- 合并由 connector 侧执行。两分支均修改 `ascend_multi_connector.py`，预期冲突区为 `_configure_layerwise_reuse_completion` 尾部：store 侧 `_configure_reuse_source_contract` 与 connector 侧 `_configure_target_provider` 均保留，判据不重叠（分别扫描 `reuse_source_contract_version` 与 `acquire_target`）。
- 若交付时间不可行，联调范围调整为栅栏全链 + 普通模式回归（两侧均已验证），receipt 全链单独补跑。

## 7. 部署

`layerwise_receipt_mode: true` 仅配置于 D 节点 kv_transfer extra config；P 节点不配置（无 provider，配置即启动失败）；P 节点回执行为由请求自带的 `request_instance` 驱动。

## 8. 验收边界与已知限制

- 已验证：27 项纯 stdlib 单元测试；ruff 0.14.0；语法编译。
- 待真机验证：`prepare_decode` 的 stream 依赖真实性；`release_target` 同步性。
- v1 限制：multi-backing 场景 wire 的 `backing_id` 发空，D 侧以租约内第一个 MAIN backing 解析；`transfer_generation` 恒 0；无独立看门狗（slot 等待以 300s 为上界）。
