# 给定物体与操作的 UMD 重跑

入口：`scripts/run_umd_given_operation.sh`。先修改脚本顶部输出目录，使用空目录启动；已有运行请先检查 `status.json`，不要重复启动。控制进程退出后，可用 `--continue` 接续同一次实验；不会自动重启失败训练。

输入固定为 2,661 张 RGB／5,232 个功能对。`test_queries.jsonl` 在推理前生成，只向模型提供 RGB、物体类别和当前操作；GT mask 由独立评分代码读取。

| 模块 | 行为 |
|---|---|
| PartGLEE | `object / part` 通用词，每张图新生成一次候选，五种 Valen 方案共享 |
| OOAL | 原最终权重，新推理；八类竞争后提取目标操作 |
| Affordance-R1 | 原 checkpoint＋官方 prompt，仅查询指定操作，输出独立 SAM2 mask |
| Valen | 候选逐个 yes/no，合并 yes；不强制存在正例，不做跨操作消重 |
| Valen SFT | 原始 Valen 初始化，同一批 76 张图；300 步头预热＋1,000 步 LoRA，固定最终权重 |

SFT 使用每个候选的原生 GT 多数类别生成二选一标签；背景参与多数统计，void 忽略。损失先在候选和同一源图的操作间平均，再对四张源图平均。双卡只改变计算分配，不改变有效 batch。

本次仍使用外部 Conda 环境：`partglee`、`valen-4b-sft`、`ooal-umd`、`affordance-r1-portable`。OOAL 与 R1 额外依赖沿用现有 baseline 配置和本地模型路径。

输出包含 `protocol.json`、`processes.json`、`status.json`、`logs/`、`candidates_train/`、`candidates_test/`、`training/`、`methods/`、`comparison/`。每个操作保存真实请求、判断分数／生成回答、二值 mask、叠加图和计时。

并行执行和保存中间文件会影响耗时；本轮记录的是离线执行成本，不将它作为严格的单卡实时延迟。所有方法完成后自动生成七方法对照页面，并更新 `docs/research/实验结果.md` 的本轮区块，保留历史结果。
