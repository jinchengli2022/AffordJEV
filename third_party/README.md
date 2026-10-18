# 上游源码

| 项目 | 固定版本 | 本地路径 |
|---|---|---|
| [PartGLEE](https://github.com/ProvenceStar/PartGLEE) | `8c6b83e0be076e8fe640943096059ee5c180de0e` | `third_party/PartGLEE/` |
| [Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) | `22366059194719ad8c4270567569e1d559491fa3` | `models/valen/` |

PartGLEE 源码直接随本仓库提供，不使用依赖外部目录的 Git submodule。上游 README、作者版权声明与源码注释保留；Valen 的 Apache-2.0 LICENSE 随模型代码保存。第三方文件遵循各自的声明。

本地适配仅涉及资源定位：

1. CLIP 通过 `MODEL.TEXT.LOCAL_MODEL_PATH` 读取仓库中的模型目录，并强制离线加载。
2. Visual Genome 两份类别词表通过源码文件位置定位，不依赖启动工作目录。
3. Detectron2 model-zoo 的配置软链接使用仓库内部相对路径。
4. 运行适配器显式从本仓库导入 PartGLEE／Detectron2，避免误用另一个安装版本。

网络结构、候选 logits、二值化、预处理和后处理数值参数未改变。原始下载仓库的 Git 元数据、旧二进制扩展不随新仓库上传；CUDA 扩展在目标环境编译。
