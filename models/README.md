# 模型文件

Git 已包含模型代码、网络配置、tokenizer、CLIP 词表；克隆后补齐下列 **5 个基础权重文件，合计约 10.44 GiB**。不要只复制 Valen 的 safetensors 而丢弃同目录配置，也不要遗漏 PartGLEE 的 CLIP。

| 文件（相对仓库根目录） | 来源 |
|---|---|
| `models/partglee/PartGLEE_SwinL.pth` | [PartGLEE 官方 Swin-L](https://drive.google.com/file/d/1D4my_0SA1f0YR7JOpC-Op92rsiaPtgST/view) |
| `models/clip_vit_base_patch32/pytorch_model.bin` | [官方安装文档指定的 CLIP](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin) |
| `models/valen/model-00001-of-00003.safetensors` | [Valen-4B 固定版本](https://huggingface.co/Valen-Team/Valen-4B/tree/22366059194719ad8c4270567569e1d559491fa3) |
| `models/valen/model-00002-of-00003.safetensors` | 同上 |
| `models/valen/model-00003-of-00003.safetensors` | 同上 |

Valen 下载示例（在仓库根目录、已安装环境中）：

```bash
conda activate valen-4b-sft
hf download Valen-Team/Valen-4B \
  --revision 22366059194719ad8c4270567569e1d559491fa3 \
  --include 'model-*.safetensors' --local-dir models/valen
```

全部文件的大小与 SHA256 见 [manifest.json](manifest.json)。放置完成后：

```bash
PYTHONPATH=src python -m affordance_jev.deployment.check_assets --sha256
```

该检查不会联网或下载；省略 `--sha256` 时只检查文件存在与大小。

## 本地微调权重

微调权重不会随 Git 上传，也不能从官方 Valen 下载得到。推理需复制相应的完整 `final/` 文件夹，允许省略仅用于恢复优化器的 `optimizer.pt`：

```text
checkpoints/valen_76/s0/321/final/                  # 历史普通 SFT
checkpoints/valen_76_abc_20261017/<方案>/final/      # s0r、a0、a1、b0、b1、c0、c1
    head.safetensors
    adapter/adapter_config.json
    adapter/adapter_model.safetensors
    training_config.json                         # A/B/C
    extra_heads.safetensors                       # B/C；C 推理必需
    status.json
```

`manifest.json` 的 `optional_checkpoints` 列出本机各方案的确切文件与校验值：

```bash
PYTHONPATH=src python -m affordance_jev.deployment.check_assets --checkpoint b0 --sha256
```

只想使用原始 Valen 时，不需要这些微调文件。想从头重新训练时，按 [训练复现](../docs/训练复现.md) 重新生成 76 张候选和决策头预热，不依赖已有 `final/`。

## 源码与环境

- PartGLEE 源码：`third_party/PartGLEE/`；CLIP 通过明确路径从 `models/clip_vit_base_patch32/` 加载。
- Valen 的官方自定义模型代码与配置：`models/valen/`，采用 `trust_remote_code=True` 的本地加载方式。
- Conda 环境位于仓库外。CUDA 扩展按目标机器重新编译，不提交 `.so`。
