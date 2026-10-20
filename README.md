# π0.5 + grounded 3D point injection

独立实现论文 [arXiv:2606.27663v2](https://arxiv.org/abs/2606.27663v2) 的三维位移条件分支。
这不是作者官方实现，也不是已经达到论文指标的模型。

**当前状态：工程初版。核心注入、数据重放/转换、训练和 ID 评估代码已编写；本地未安装 JAX/CUDA/MuJoCo，尚未完成 GPU、重放或 rollout 验收。LIBERO-PRO 的 Task/Position 扰动适配未完成，不能声称已完整复现论文。**

## 同学接手后先做什么

1. 阅读 [交接说明](docs/handoff.md) 和 [验收状态](docs/validation.md)。
2. 在 Linux NVIDIA GPU 机器上 clone 本仓库；不用接收数据或模型压缩包。
3. 准备 `git` 和 `uv`，执行环境安装及小测试。先回传 `runs/reports/` 和小测试日志，再安排长时间训练。
4. 数据、权重均有下载脚本。所有运行输出都在本仓库相对目录 `runs/` 下。

```bash
bash scripts/setup.sh
bash scripts/setup_simulator.sh
bash scripts/smoke_test.sh 2>&1 | tee runs/logs/smoke.log
```

训练环境沿用官方 OpenPI 的 `pyproject.toml` 和 `uv.lock`，位于 `openpi/`。
仿真重放使用单独的 Python 3.10 环境，避免旧 LIBERO 与训练依赖互相冲突。
不需要注册日志服务账号，默认关闭 WandB 远端上传。

## 数据和权重

```bash
source scripts/env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py weights
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py data --scale 69
```

权重使用 `pi05_base`，不使用已经微调过的 `pi05_libero` 作为主实验初始化。
原始数据从 LIBERO 官方指向的 Hugging Face 仓库下载，并记录下载 revision。
详细链接和网络失败时的一般方法见 [数据与权重](docs/data_and_weights.md)。

下载后需要重新渲染深度、mask 和相机几何，不能拿普通 RGB LeRobot 数据直接当空间分支数据。

```bash
bash scripts/prepare_data.sh 69
```

该命令会下载、重放、转换和计算两组归一化统计。已存在的重放/转换输出不会被自动覆盖。
只需重做某一步时，按该脚本中的单步命令操作；请勿反复运行整个 prepare 脚本。
首次建议先检查少量重放帧：

```bash
runs/env/simulator/bin/python scripts/replay_dataset.py --scale 20 --max-tasks 1 --max-demos-per-task 1
```

这个部分重放不能作为正式 20 任务数据转换，程序会检查任务数量。
小测试与正式重放使用不同 scale 的目录，或在单独的工作副本里做小测试。
重放的坐标轴、RGB 方向、状态/动作时序必须经远端核对；发现空 mask 或缺失状态时默认报错，没有质心/零向量替代。

## 配对训练

配置分别在 `configs/baseline.json` 和 `configs/spatial.json`，默认两组均为 batch 256、30,000 steps、chunk 10、seed 42。
AdamW、峰值 LR 5e-5、10,000 步 warmup 后恒定、weight decay 1e-10、clip norm 1.0。
原基座结构、原 flow-matching loss 不变；空间组新增 `3→1024→1024` ReLU MLP，最后一层权重和 bias 均零初始化。
输出添加到 π0.5 的时间条件，复用原 adaRMSNorm；不重写成另一种 LayerNorm。

```bash
bash scripts/train_baseline.sh --repo-id local/pi05_3d_69 --fsdp-devices 1
bash scripts/train_spatial.sh --repo-id local/pi05_3d_69 --fsdp-devices 1
```

多卡模型切分时把 `--fsdp-devices` 改为可用设备数的合法因子，并确保 global batch 能整除总设备数。
不能因为用了多卡就给两组设置不同的 global batch。
恢复训练给原命令加 `--resume`；新实验使用新的 `--exp-name`。
没有自动启用覆盖旧 checkpoint。

确认小测试和数据检查通过后，可顺序执行准备与两组训练：

```bash
bash scripts/run_all.sh 69 1
```

第二个数字是 FSDP 设备数。该入口不是“完整论文复现一键通过”：训练结束后仍需评估。
任一步失败会停止，退出码和阶段名写入 `runs/reports/`，完整日志写入 `runs/logs/`。

## 评估和回传

训练结果、日志、环境报告、下载记录均保存在 [runs/](runs/README.md)。
ID 策略服务和 rollout 示例见 [评估协议](docs/evaluation_protocol.md)。
**Task/Position 两种 OOD 评估适配尚未完成，程序会明确拒绝，而不是运行 ID 冒充 OOD。**

你可以把 `runs/reports/`、`runs/logs/`、`runs/evaluation/` 回传给项目负责人；这些目录默认不进 Git。
如果需要一个方便回传的结果 ZIP，可选执行：

```bash
runs/env/openpi/bin/python scripts/package_results.py
```

这个命令只打包小结果，不含数据、预训练权重或训练 checkpoint。
需要训练权重时额外加 `--include-checkpoints`；体积可能很大。
代码通过 GitHub 仓库协作，不再生成代码 ZIP。

## 来源与许可证

- OpenPI 固定 commit：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- LIBERO 固定 commit：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 完整来源：`UPSTREAM.json`；对上游 6 个文件的修改：`docs/openpi_changes.patch`。
- 上游 OpenPI 原始 Apache 2.0 和 Gemma 许可证保留在 `openpi/`；论文源文件属于原作者。
- 阅读笔记见 `papers/reading_notes_v2.md`，实现选择与论文未报告内容见 [差异清单](docs/reproduction_spec.md)。
