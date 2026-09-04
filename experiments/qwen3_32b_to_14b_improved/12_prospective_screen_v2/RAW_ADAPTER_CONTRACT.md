# Real GPU runner contract

`scripts/run_gpu.py` 是本目录内的真实、fail-closed、可恢复执行入口，不是只接收外部结果的依赖注入
壳。当前由于 10 尚未最终发布，template placeholder 会阻止入口运行；代码完成不等于已启动 GPU。

正式入口严格执行：

1. 重建 formal config、prepared profile 与 content-addressed run definition；
2. 要求固定 venv interpreter、`PYTHONDONTWRITEBYTECODE=1`、精确的
   `PYTHONPYCACHEPREFIX`（启动前必须不存在，包括 dangling symlink）与
   `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，获取全局 8-GPU flock；bytecode policy 在任何本地 `src`
   import 前 fail-closed；
3. 在 import torch/CUDA 之前，深度 SHA-256 复验 32B/14B 所有 weight shards、mapper、02/03
   interfaces 和 upstream runtime code closure；
4. 通过 nvidia-smi 连续 idle gate 后，精确复验 Python/distribution、torch/CUDA 版本与 8 卡可见性；
5. 真实加载 Qwen3-32B 到卡 0..4（层数 `12,13,14,13,12`），Qwen3-14B 到卡 5..7
   （`13,14,13`），并持久化每层实际设备检查；
6. 构造 native14、NVIDIA k20、02 shrinkage、03 tail-refresh R32 四个真实 engine；R32 必须共享同一个
   k20 builder，不能加载替代 mapper；
7. 每个 sample 按 `native14 → nvidia_k20 → shrinkage → tail_refresh_r32` 写原子 raw result。NVIDIA
   绑定 native SHA，两个 improved mode 同时绑定 native 与 NVIDIA SHA；
8. 使用本目录 audited greedy loop 持久化 exact generated token IDs/hash、decoded continuation、
   stop reason/cap，以及模型实际观察但不计入 decoded IDs 的 `terminal_token_id`。EOS 时该值必须
   属于 `frozen_eos_token_ids`；集合由 runtime-provenance 已内容寻址的 target `config.json` 与
   `generation_config.json` 精确并集得到，并在模型加载后与 tokenizer/model runtime 值逐项相等。
   stop-sequence/max-tokens 的 terminal 必须为 null。成本字段固定为 monolithic wall time、精确 GPU
   keys `0..7` 的 per-GPU peak allocation；phase timing 只能是 unavailable/null 和唯一 monolithic
   scope，绝不伪造；
9. R32 audit 必须 exact 重算 builder/full prompt lengths、mapped/native slices、31 个 position/cache
   positions、attention-mask shape、forward/query counts 与 cache before/after；所有整数使用拒绝 bool 的
   `strict_int`，所有嵌套 JSON 使用递归类型精确比较，不能只验计数和式或依赖 Python `True == 1`；
10. 单格失败写 `.failure.json` 并停止；重启时只复用语义和 sidecar 全部通过的格。若进程死在 result
    原子 rename 与 sidecar rename 之间，会先完整重验 result，再补 sidecar；若死在成功结果发布与
    stale failure 清理之间，resume 仅在完整语义验证后清理该 marker；
11. 每个 raw cell 写入其 `execution_attempt_id`；跨进程 resume 后，completion ledger 会验证所有实际
    贡献 cell 的 attempt，而不只验证最后一次启动；所有 attempt counts 同样使用 `strict_int`。四模式
    raw grid 完整后，写入绑定每个贡献 attempt
    的 idle/deep-code/CUDA/method/topology evidence 的 `raw_completion_receipt.json`。发布 receipt 前会用
    frozen tokenizer 全量重验 token decode、metric、cross-mode SHA、R32 audit 和 exact raw JSON tree；
    receipt-sidecar kill window 也只能在同一套深验通过后恢复。没有该 receipt，`seal_run.py` 拒绝任何
    人工或 CPU-only raw grid；
12. GPU runner 与 `seal_run.py` 共用 per-run flock；GPU 在锁内重查 raw/terminal marker，等待的第二
    runner 不能再次加载模型。terminal status 是不可降级的单向 CAS，seal 后 GPU failure/live status
    不能覆盖。`seal_run.py` 在 CPU 上生成离线 two-pass、summary、decision、completion receipt；
    `validate_run.py` 从冻结输入独立重算全部内容。

样本数由 formal config 精确绑定 10 最终 publication；每个 profile 的 raw cells=`samples×4`，derived
cells=`samples×1`。smoke 固定 4 samples/16 raw/4 derived。screen/confirm 在 10 component-feasibility
完成前不在本模板内冻结。

runner 不 import 07。07 只是 `reference_interfaces.json` 中内容寻址的设计参考；02 shrinkage、03 R32、
NVIDIA runtime/model/mapper/code closure 都分别由 formal config 和 upstream receipts 的 path/nbytes/SHA
复验。
其中 `kvcache_transfer/__init__.py` 与 `kvcache_transfer/io.py` 会由 config 的独立
`runtime_package_closure` 逐字节复验；它们不能因历史 upstream receipt 未列出而脱离 provenance。
