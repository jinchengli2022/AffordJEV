# Valen：76 张训练图的 A／B／C 对照

所有配置均为固定最终 1,000 步，seed=321、有效源图 batch=4，复用同一份 300 步头预热权重。测试不参与训练或权重选择。

| 配置 | 方法 |
|---|---|
| `s0r.json` | 单卡梯度累计的普通 SFT 对照 |
| `a0.json` / `a1.json` | 相同显示／定义增强；A1 增加区域与定义配对、预测一致性 |
| `b0.json` / `b1.json` | 相同几何变体、组成监督；B1 增加面积加权组成一致性 |
| `c0.json` / `c1.json` | 相同残差头；C0 用目标区域，C1 增加最近邻与整物体特征 |

环境：项目外 Conda `valen-4b-sft`。`records` 为原 76 张训练记录；`views` 仅为这些图片的变体。每份 JSON 明示模型、输入、预热权重、输出目录及损失权重。

代码：`src/affordance_jev/experiments/umd_valen/abc/`。`data.py` 生成训练变体，`train.py` 训练，`run_suite.py` 串联每组训练→推理→评测，`report.py` 汇总结果。推理复用现有 `umd_valen.infer`；B 丢弃辅助头，C 加载 `extra_heads.safetensors`。

重复实验须先为 JSON 中的输出／权重及 suite 目录填写新路径；程序拒绝覆盖已存在的实验。训练及测试流程日志保存在各输出目录。
