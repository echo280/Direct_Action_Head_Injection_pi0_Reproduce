# 数据和权重的来源

## 模型

- 官方代码：https://github.com/Physical-Intelligence/openpi
- 主实验初始化：`gs://openpi-assets/checkpoints/pi05_base`
- 工程下载入口：`runs/env/openpi/bin/python scripts/download.py weights`
- 文件落点：`runs/weights/openpi-assets/checkpoints/pi05_base/`。

脚本复用官方 `maybe_download`，保持完整目录结构。网络不通时，可在另一台可访问官方对象存储的机器上用同一脚本下载，保留目录结构后同步到上述路径。不要使用来源不明的同名权重。

官方示例 `pi05_libero` 是已经训练过的模型，不能替代 `pi05_base` 作为本文配对微调的共同起点。

## 数据

- 官方 LIBERO：https://github.com/Lifelong-Robot-Learning/LIBERO
- 官方下载工具指向的数据集：https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- 原始下载方法：https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/benchmark_scripts/download_libero_datasets.py
- 工程下载入口：`runs/env/openpi/bin/python scripts/download.py data --scale 69`
- 69 任务名单：`configs/tasks_69.json`；20 任务名单：`configs/tasks_20.json`。

69 任务是 Object 10 + Spatial 10 + 论文附录精确列出的 LIBERO-90 49。下载工具兼容数据仓库中的 `libero_90` / `libero_100` 目录与 ZIP 命名；重放按任务名定位 HDF5，缺文件直接失败。

下载脚本在运行时查询一次 Hugging Face revision，再固定该 revision 下载，写入 `runs/reports/download_data.json`。请保留该记录。

论文所需几何由原始 HDF5 的 `states` 和每 demo 的 `model_file` 重放恢复。只含 RGB、动作和普通 state 的转换数据不足以恢复相机可见深度与 mask。

如果对应数据镜像不提供这些字段，需要从官方渠道取得带状态的原始演示；不能把公开 3D 质心直接当本方法输入。脚本会报错，需要先解决数据来源。

## 存储

预留原始数据、重放 RGB episode、LeRobot 图像以及两组训练 checkpoint 的空间。
不在不了解任务长度和 checkpoint 保留策略时承诺固定磁盘体积；用 `runs/reports/environment.json` 中的磁盘报告检查空间。
数据、权重、缓存、环境和结果都放在 `runs/`，这些大文件不会推送到 GitHub。
