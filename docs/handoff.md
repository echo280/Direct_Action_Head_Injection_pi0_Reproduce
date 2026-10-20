# 交接说明

目标：在同一份 LIBERO 数据上，从同一个 `pi05_base` 权重分别训练 π0.5 baseline 和加入三维位移条件的 spatial 模型，再比较评估结果。仓库名保留了 `pi0`，实际使用的是 **π0.5**。

**当前是待 GPU 验收的工程初版。数据重放、真实权重加载、训练和仿真评估均未在本地跑通；目前提供普通 LIBERO 的 ID（原始任务分布）评估入口，论文需要的 LIBERO-PRO Task/Position OOD（分布外扰动）评估尚未实现。** 下面给出的是实际已有入口，不能把跑完这些命令等同于论文完整复现。

## 1. 执行顺序和脚本位置

可以按“获取数据 → 训练 → 评估”推进，但首次运行还需要安装环境；数据下载后也必须重放、生成几何条件并转换格式。smoke 是正式运行前的小检查，不是一份需要另外下载的数据，也不需要每一步等待项目负责人确认。检查通过就继续，失败时保留日志并反馈。

| 阶段 | 仓库中的入口 | 做什么 |
| --- | --- | --- |
| 训练环境 | [`scripts/setup.sh`](../scripts/setup.sh) | 按 OpenPI 的锁文件安装训练依赖，记录机器和软件信息 |
| 仿真环境 | [`scripts/setup_simulator.sh`](../scripts/setup_simulator.sh) | 下载固定版本 LIBERO，安装独立 Python 3.10 重放/评估环境 |
| 小检查（smoke） | [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) | 运行几何单元测试和小模型接口检查 |
| 下载基座权重 | [`scripts/download.py`](../scripts/download.py) 的 `weights` 命令 | 下载 `pi05_base` |
| 数据准备 | [`scripts/prepare_data.sh`](../scripts/prepare_data.sh) | 下载数据 → 仿真重放/生成三维位移 → 转为 LeRobot → 计算两组归一化统计 |
| baseline 训练 | [`scripts/train_baseline.sh`](../scripts/train_baseline.sh) | 不使用空间条件的对照组 |
| spatial 训练 | [`scripts/train_spatial.sh`](../scripts/train_spatial.sh) | 使用三维位移条件的实验组 |
| 策略服务 | [`scripts/run.py`](../scripts/run.py) 的 `serve` 命令 | 加载训练 checkpoint，接收模拟器观测并输出动作 |
| ID 评估 | [`scripts/evaluate.py`](../scripts/evaluate.py) | 连接策略服务，运行 LIBERO Object/Spatial 回合 |

除首次 clone 外，下文命令都在**仓库根目录的 Linux Bash 终端**运行。无需手动激活 conda/venv；命令已经指定对应环境的 Python。每次打开新终端，都先进入仓库并运行 `source scripts/env.sh`。下载、缓存、项目虚拟环境、checkpoint、日志和指标统一放在仓库相对目录 `runs/` 中。

## 2. 取代码和准备基础工具

使用 Linux NVIDIA GPU 机器，先确认机器已有可用的 NVIDIA 驱动：

```bash
nvidia-smi
git --version
curl --version
```

缺少 `git`、`curl` 时，在有 sudo 权限的 Ubuntu/Debian 机器上安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl
```

缺少 `uv` 时，使用 [uv 官方安装方法](https://docs.astral.sh/uv/getting-started/installation/)；下面的第一条命令会执行其官方安装脚本。安装后重新打开终端，再确认 `uv` 可用：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv --version
git clone https://github.com/echo280/Direct_Action_Head_Injection_pi0_Reproduce.git
cd Direct_Action_Head_Injection_pi0_Reproduce
source scripts/env.sh
git rev-parse HEAD > runs/reports/code_commit.txt
```

代码已经拉取过时，在开始新实验前执行 `git pull --ff-only`，然后重新记录提交号。若本地有代码修改，先保留并处理修改，不要强制覆盖。长任务应在机器已有的持久终端会话或作业调度器里运行，避免断开连接时任务退出。

## 3. 安装两个环境并检查

```bash
source scripts/env.sh
set -euo pipefail
bash scripts/setup.sh 2>&1 | tee runs/logs/setup_training.log
bash scripts/setup_simulator.sh 2>&1 | tee runs/logs/setup_simulator.log
bash scripts/smoke_test.sh 2>&1 | tee runs/logs/smoke.log
```

`set -euo pipefail` 使这里的命令失败时停止，也避免 `tee` 掩盖错误。依赖安装可能需要访问 GitHub、Python 包源和 PyTorch 下载源。

- 训练环境：`runs/env/openpi/`。依赖定义及锁文件在 [`openpi/pyproject.toml`](../openpi/pyproject.toml)、[`openpi/uv.lock`](../openpi/uv.lock)，安装脚本使用 `uv sync --frozen`。
- 仿真环境：`runs/env/simulator/`。依赖在 [`configs/requirements_sim.txt`](../configs/requirements_sim.txt)，LIBERO 源码放在 `runs/cache/LIBERO/`。两个环境分开，避免训练依赖与旧版仿真依赖冲突。
- 环境报告：`runs/reports/environment.json`；仿真依赖实际版本：`runs/reports/simulator_requirements_resolved.txt`。

**smoke 具体在哪里、检查什么？** `scripts/smoke_test.sh` 依次运行 [`tests/test_geometry.py`](../tests/test_geometry.py) 和 [`scripts/smoke_model.py`](../scripts/smoke_model.py)。前者检查几何计算、阶段切换和缓存对齐；后者用随机初始化的小模型检查空间分支零初始化、梯度、原时间条件/初始 loss 一致，以及缺失空间输入会报错。

通过时命令退出码为 0，且 `runs/reports/model_smoke.json` 中有 `"status": "passed"`。这一步不下载预训练权重，不进行正式训练，也不验证 MuJoCo 渲染。EDT 单元测试在缺少 scipy 时会跳过；正式仿真环境仍需要完整依赖。

正式训练还应确认 JAX 能识别 GPU，避免小模型在 CPU 上通过检查后误启动长训练：

```bash
runs/env/openpi/bin/python -c 'import jax; devices = jax.devices(); print(devices); assert any(d.platform == "gpu" for d in devices), "JAX did not detect a GPU"' 2>&1 | tee runs/logs/jax_devices.log
```

如果失败，先提供对应安装日志、`environment.json` 和命令报错，修复后从失败步骤继续。重试前先归档旧日志，避免同名 `tee` 命令覆盖它们。不要直接忽略错误进入训练。

## 4. 获取权重、获取并处理数据

默认先做 **69 任务**主实验，两组共用同一份准备好的数据：

```bash
source scripts/env.sh
set -euo pipefail
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py weights 2>&1 | tee runs/logs/download_weights.log
bash scripts/prepare_data.sh 69 2>&1 | tee runs/logs/prepare_data_69.log
```

`prepare_data.sh` 已经包含数据下载，无需在它前面重复执行下载命令。它实际按下面顺序执行；**以下仅用于定位失败步骤，上一段成功后不要再重复执行这一段**：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py data --scale 69
runs/env/simulator/bin/python scripts/replay_dataset.py --scale 69
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/convert_dataset.py --scale 69
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run.py norm baseline --repo-id local/pi05_3d_69
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run.py norm spatial --repo-id local/pi05_3d_69
```

普通 RGB 数据没有本方法需要的三维位移条件，因此不能下载完就直接开始空间组训练。重放步骤利用模拟器状态和几何信息生成目标点、目标/夹爪坐标及位移；转换步骤生成训练数据和逐帧对齐的几何缓存。

| 产物 | 相对路径 |
| --- | --- |
| 初始权重 | `runs/weights/openpi-assets/checkpoints/pi05_base/` |
| 原始 LIBERO 数据 | `runs/data/raw/` |
| 重放后的逐演示 NPZ | `runs/data/replayed_69/` |
| LeRobot 训练数据 | `runs/data/lerobot/local/pi05_3d_69/` |
| 三维位移缓存及元信息 | `runs/data/geometry/local/pi05_3d_69/cache.npz` 和 `cache.json` |
| 两组归一化统计 | `runs/assets/pi05_3d_baseline/`、`runs/assets/pi05_3d_spatial/` 下对应数据集目录 |
| 下载/转换记录 | `runs/reports/download_weights.json`、`download_data.json`、`data_conversion.json` |

下载来源和网络失败后的替代方法见 [数据与权重](data_and_weights.md)。不需要把训练数据和权重提交到 GitHub。

重放完成后，应检查 RGB 方向、目标点位置、坐标系及状态/动作对齐是否合理。目前没有自动完成这些视觉核对的验收脚本；程序不报错也不等于这些选择都已正确。空 mask、缺失状态或几何错误应作为问题反馈，不能用零向量替代后继续。

**重放和转换不支持自动断点续跑，会拒绝覆盖现有输出。** 失败后先保留报错和相关产物，确认失败位置；已经完成重放、仅转换或统计失败时，可以从对应的单步命令继续。重放/转换本身中途失败时，需要处理部分输出或使用新的工作目录，不能反复运行整个准备脚本。20 任务属于另一项实验，不要把小规模数据混入 69 任务结果。

## 5. 训练 baseline 和 spatial

```bash
source scripts/env.sh
set -euo pipefail
bash scripts/train_baseline.sh --repo-id local/pi05_3d_69 --fsdp-devices 1 2>&1 | tee runs/logs/train_baseline.log
bash scripts/train_spatial.sh --repo-id local/pi05_3d_69 --fsdp-devices 1 2>&1 | tee runs/logs/train_spatial.log
```

两组顺序训练，均从同一个 `pi05_base` 初始化；不要把训练后的 baseline checkpoint 当作 spatial 的初始权重。配置在 [`configs/baseline.json`](../configs/baseline.json) 和 [`configs/spatial.json`](../configs/spatial.json)，默认均为 global batch 256、30,000 steps、action chunk 10、seed 42。

`--fsdp-devices 1` 表示不把单份模型分片到多张卡，并非限制只使用一张 GPU。若需要模型分片，把它改为可见设备数的合法因子，并保持 global batch 能被总设备数整除；两组使用相同设置。不要只给一组改 batch 或步数。

| 产物 | 相对路径 |
| --- | --- |
| baseline checkpoint | `runs/checkpoints/pi05_3d_baseline/paper_seed42/<step>/` |
| spatial checkpoint | `runs/checkpoints/pi05_3d_spatial/paper_seed42/<step>/` |
| 训练参数记录 | `runs/reports/baseline_paper_seed42_train.json`、`spatial_paper_seed42_train.json` |

30,000 steps 的最后保存目录通常为 `29999`，评估前以实际目录为准。训练中断后，在原训练命令中加 `--resume` 恢复；新实验加 `--exp-name 新名称`，使用新的日志文件名。恢复时继续使用原来的数据、配置和实验名。

## 6. 训练后进行 ID 评估：两个终端

服务端负责模型推理，客户端负责模拟器回合；两者在同一台机器上，通过端口 8000 通信。两个终端都先进入仓库根目录。以下 checkpoint 路径中的 `29999` 要替换为实际保存的步目录。

**先评估 baseline。终端 A 启动模型服务，并保持运行：**

```bash
source scripts/env.sh
set -euo pipefail
runs/env/openpi/bin/python scripts/run.py serve baseline --repo-id local/pi05_3d_69 \
  --checkpoint runs/checkpoints/pi05_3d_baseline/paper_seed42/29999 --port 8000 \
  2>&1 | tee runs/logs/serve_baseline.log
```

**终端 B 等服务启动并监听端口后执行：**

```bash
source scripts/env.sh
set -euo pipefail
runs/env/simulator/bin/python scripts/evaluate.py --variant baseline --suite libero_object --trials 50 --port 8000 \
  2>&1 | tee runs/logs/eval_baseline_object.log
runs/env/simulator/bin/python scripts/evaluate.py --variant baseline --suite libero_spatial --trials 50 --port 8000 \
  2>&1 | tee runs/logs/eval_baseline_spatial.log
```

**baseline 两项完成后，在终端 A 按 Ctrl+C 停止旧服务，再启动 spatial 服务：**

如果终端因启用了 `set -e` 而退出，重新打开终端并进入仓库，再执行下面的命令。

```bash
source scripts/env.sh
set -euo pipefail
runs/env/openpi/bin/python scripts/run.py serve spatial --repo-id local/pi05_3d_69 \
  --checkpoint runs/checkpoints/pi05_3d_spatial/paper_seed42/29999 --port 8000 \
  2>&1 | tee runs/logs/serve_spatial.log
```

**终端 B 等新服务就绪后执行：**

```bash
source scripts/env.sh
set -euo pipefail
runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite libero_object --trials 50 --port 8000 \
  2>&1 | tee runs/logs/eval_spatial_object.log
runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite libero_spatial --trials 50 --port 8000 \
  2>&1 | tee runs/logs/eval_spatial_spatial.log
```

**服务端 checkpoint 与客户端 `--variant` 必须一致。** 每个 suite 有 10 个任务，每任务 50 回合，即每个模型共 1,000 个 ID 回合。两组默认使用相同初始状态和 seed。

结果在 `runs/evaluation/`，例如 `baseline_libero_object_id.jsonl` 保存逐回合记录，`baseline_libero_object_id.summary.json` 保存成功率及错误数；其他三项命名对应替换。检查 summary 的 `errors` 是否为 0；推理/仿真错误不能直接解释为模型能力差。已有同名 JSONL 时程序会拒绝混写，重新评估前先归档原结果。

这些命令只做 ID。**不能通过加 `--condition task` 或 `--condition position` 获得论文 OOD 结果，目前这两个选项会报未实现。** 完整论文复现仍需补齐并验证 LIBERO-PRO 适配，见 [评估协议](evaluation_protocol.md)。

## 7. 已理解步骤后，如何少敲命令

对于尚未产生重放/转换输出的新工作目录，安装好 `git`、`curl`、`uv` 并取代码后，可用现成总入口代替第 3～5 节的逐步命令：

```bash
bash scripts/run_all.sh 69 1
```

它依次安装两套环境、运行 smoke、下载权重、准备数据、训练 baseline、训练 spatial，任一步失败就停止。第二个数字是 FSDP 设备数。它**不包含第 3 节额外的 JAX GPU 断言，也不包含第 6 节评估**；首次机器验收建议按分步命令确认 GPU 可用。不要在已经完成数据准备后再从头运行总入口。

总日志为 `runs/logs/pipeline_<UTC时间>.log`；阶段和退出码为 `runs/reports/pipeline_<UTC时间>.status`。训练完成后按第 6 节评估。现阶段可以减少手工命令，但还不能称为“一条命令完成论文所有实验”。

## 8. 如何交回结果和反馈问题

项目负责人不能远程登录，所以请通过双方可访问的文件传输方式交回：

- `runs/reports/`：代码版本、环境、下载 revision、配置和阶段状态。
- `runs/logs/`：安装、检查、数据准备、训练和评估日志。
- `runs/evaluation/`：逐回合结果和汇总指标。

训练 checkpoint 保留在算力机器的 `runs/checkpoints/`，需要传模型时再单独同步选定步目录。以上运行产物默认被 Git 忽略，`git push` 不会自动上传它们；代码修改通过分支和提交同步。

如需方便回传的小结果包，可执行 `runs/env/openpi/bin/python scripts/package_results.py`，默认不包含训练数据、预训练权重和 checkpoint。这是可选的结果打包，与代码通过 GitHub 交接并不冲突。

失败时附上完整命令、`git rev-parse HEAD`、对应日志和报告；数据错误附任务名/demo/帧索引。先定位、修复失败步骤，再继续后续阶段，无需每阶段人为等待确认。不要发送密码、访问令牌或个人账号文件。
