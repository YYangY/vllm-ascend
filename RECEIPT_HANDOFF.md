# RECEIPT_HANDOFF（connector side → store side）

- 分支：`feat/layerwise-receipt` @ `e26962a`（仓库 `YYangY/vllm-ascend`，基线 `4234a1f1`，与你的 `p-main-buffer-reuse` 同基线）
- 接口准绳：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_contract.py`（类型 + TargetProvider 协议 + 校验函数，注释自包含，照它实现即可）
- 参考实现：同目录 `fake_target_provider.py`（四个方法的调用形态可以照抄）

---

## 1. 本侧两个提交做了什么

### `a5581b9` — receipt 账本 + 逐层终态 + 槽位复用栅栏（+2658 行，7 文件）

| 文件 | 内容 |
|---|---|
| `mooncake_layerwise_contract.py` | 共享契约：RequestKey / BlockAllocation / TargetLease / ReceiveReceipt 等类型，TargetProvider 四方法协议，启动能力断言 + 逐请求租约校验。纯 stdlib，CPU 可跑 |
| `mooncake_layerwise_receipt.py` | 双账本：**ReceiptLedger**（D 侧，按七字段身份记账判就绪，不数消息个数）+ **SlotLedger**（P 侧，槽位代次 + 票据的复用栅栏，三支等待判定） |
| `mooncake_layerwise_connector.py` | LAYER_RESULT/RANK_TERMINAL 逐层回执协议（十四字段 wire，receipt 请求的 DONE 带 5 元组版本号分流，防提前放行）；12 个插入点 |
| `ascend_multi_connector.py` | provider 注入三支判定：没开 receipt 模式且没有 provider → 跳过；开了没有 → 拒绝启动；多于一个 → 拒绝启动 |
| `fake_target_provider.py` + `tests/ut/kv_offload/` | 可脚本化的测试替身 + 27 条纯 stdlib UT（`python tests/ut/kv_offload/test_layerwise_receipt.py` 本机可复跑） |

### `e26962a` — 跨进程运输 + 延迟释放 + 注释自包含（+297/−236，7 文件）

1. **receipt spec 改走 connector metadata**：调度进程把账本规格（req_id / 期望 rank 数 / backings / prompt 长度 / lease）装进 metadata，D worker 收到后自己建账本。为什么：vLLM 标准部署里调度进程和 worker 是两个进程，同进程内直接递对象在真机上静默失效（本机单进程 UT 测不出这个坑）。
2. **`request_instance` 写入本请求的 kv_transfer_params**：worker 不用额外通道就能拿到身份。
3. **请求结束调 `release_target`**；失败时保留 lease 并**延迟释放块**（provider 判不安全释放时，立即释放会把可能仍被引用的内存给下一个请求 → 脏写）。
4. **修复 `backings` 未定义**：发布路径现在从 lease 的 MAIN 视图绑定的 region 推导 MAIN backing 列表（此前该变量从未被定义过——纯 stdlib UT 不覆盖调度侧路径，所以之前没炸出来）。
5. 注释/docstring 全部自包含，ruff 0.14.0 全绿。

## 2. 还需要什么（store side 唯一缺口）：TargetProvider

按契约文件实现四方法 + 三属性：

```python
contract_version = 1
visibility_contract = "stream-dependency"
supports_layerwise_buffer_reuse = True
acquire_target(request, allocation) -> TargetLease   # 覆盖全 prompt，不足 raise，不返半套
get_target_views(lease) -> tuple[CacheView, ...]     # 平铺；消费方按 view.component 分组
prepare_decode(lease, receipt) -> None               # 正常返回 = 消费 stream 依赖已接
release_target(lease) -> None                        # 正常返回 = 引用已同步释放
```

挂载位置：D 节点的 connector 或 worker 上、暴露 `acquire_target` 即可（收集器两个都扫；注入时会跑能力断言）。

## 3. 为什么必须是你来做、为什么缺它不行

- **为什么必须是 store side 实现**：`acquire_target` 分配的是 store 侧 KV 空间的租约，`release_target` 的"引用归零"是 store 侧运行时才有的事实——connector 侧既造不出也替代不了（FakeTargetProvider 只够 UT 用）。
- **为什么缺它连启动都过不去**：D 节点开了 `layerwise_receipt_mode` 后，启动时扫不到 `acquire_target` 会直接拒绝启动。这是故意的：宁可拒绝启动也不半套上线。
- **为什么缺它收齐了也不放行 decode**：链路最后一环是 `prepare_decode`——`visible` 是它的**产出**而不是前提。D 侧数据齐了会调它，它正常返回才置 visible、才放行 decode；它不存在，D 收齐所有层也在设计上永远等下去。
- **为什么 TargetLease 必须纯数据可 pickle**：lease 装 in receipt spec 里经 connector metadata 跨进程序列化，塞运行时对象（锁、句柄）会炸。
- **为什么 release_target 要同步返回**：release 返回后同地址可能立刻被下一请求复用重写；在途引用没清干净就是脏读。

## 4. 交付时顺带回答（联调首日要验的三件事）

1. `prepare_decode` 是真接了消费 stream 依赖，还是空壳？
2. `acquire_target` 容量不足抛什么异常、有无部分成功路径？（本侧假设：异常 = 整请求失败）
3. `release_target` 是否同步返回（返回后引用已清）？

## 5. 范围划界与合并

- **五方法源接口（open_source / source_ready 等）本侧下一轮再接**：本次 receipt 链路不经过它；你的自动安装发现没有 version-1 consumer 会静默跳过，不报错，不用等本侧。
- **合并由本侧做**：两个分支都动过 `ascend_multi_connector.py`，预期冲突区在 `_configure_layerwise_reuse_completion` 尾部——你的 `_configure_reuse_source_contract` 和本侧的 `_configure_target_provider` 都保留即可，判据不重叠（一个扫 `reuse_source_contract_version`，一个扫 `acquire_target`）。
- 明早交不了就明说：本侧按 Fallback 跑（栅栏全链 + 普通模式回归，两边已绿），receipt 全链单补一轮，不影响明天开工。

## 6. 部署与已知边界（v1）

- 开关：`layerwise_receipt_mode: true` **只在 D 节点**的 kv_transfer extra config 里开；P 节点不开（P 没有 provider，开了会拒绝启动），P 的回执行为由请求自带 instance 驱动。
- 已知 v1 边界（都有兜底，不影响首跑）：multi-backing 场景 wire 的 backing_id 发空、D 用自己 lease 的第一个 MAIN backing 兜底；`transfer_generation` 恒 0；无独立看门狗（slot 等待有 300s 上界兜底）。
