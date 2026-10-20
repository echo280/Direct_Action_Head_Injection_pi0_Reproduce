# 统一运行目录

所有本工程生成的数据、权重、缓存和实验结果写入此目录及其子目录。
训练与统计：`assets/`、`checkpoints/`；日志：`logs/`；指标：`evaluation/`；状态与报错：`reports/`。
下载与数据准备：`data/`、`weights/`、`cache/`；Python 环境：`env/`。

这些大文件不会提交进代码仓库。回传给项目负责人的小结果至少包含 `reports/`、`logs/` 和 `evaluation/`。
训练权重保留在本机；需要共享时单独传输 `checkpoints/` 中选定的检查点。
