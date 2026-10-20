# ID / Task / Position 评估协议

代码已接入两类 OOD，**尚未通过真实 MuJoCo/GPU rollout 验收，也没有论文指标结果**。配置入口是 [`configs/evaluation.json`](../configs/evaluation.json)。

## 固定来源和实际映射

| 条件 | 后端 | Object / Spatial 注册任务集 |
| --- | --- | --- |
| ID | 固定版本原始 LIBERO | `libero_object` / `libero_spatial` |
| Task OOD | 固定版本官方 LIBERO-PRO | `libero_object_task` / `libero_spatial_task` |
| Position OOD | 固定版本官方 LIBERO-PRO | `libero_object_swap` / `libero_spatial_swap` |

- [LIBERO-PRO 官方代码](https://github.com/Zxy-MLlab/LIBERO-PRO)，固定 commit `eafdb809426b13153aa1e4c42d6601844217dfec`。
- [官方预生成 BDDL / 初始状态](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)，固定 revision `c86fc3b8293185a6f373677018ff3e37f8391602`。
- 下载四个 OOD suite 的 40 个 BDDL 和 40 个 `.pruned_init` 文件；不现场重新随机生成扰动，不启用 Object/Language/Environment 等额外扰动。
- Task 指令读取实际环境的 `language_instruction`，抓取对象和放置目标读取该 BDDL 的 goal。官方文件名可能保留旧对象名称，不能拿文件名生成指令。
- Position 使用官方 `_swap` BDDL 和与之配套的初始状态；不在原始 ID 状态上随意加噪声。
- 每回合用新任务的对象高度初始化阶段切换；升高超过 1 cm 后切到新任务的放置目标。三维位移仍由可见像素、深度和标定计算。

目标论文没有公开精确 LIBERO-PRO commit、预生成文件 revision 和全部控制细节。因此本版明确锁定上述公开版本；Object/Spatial horizon 为 280/220，replan 为 5 步，初始化稳定 10 步，seed 为 7。它们是可审计的复现选择，不能声称已经与作者私有评估设置逐项相同。

## 1. 下载 OOD 资源

先完成交接说明中的训练和仿真环境安装，在仓库根目录执行：

```bash
source scripts/env.sh
set -euo pipefail
bash scripts/setup_ood.sh 2>&1 | tee runs/logs/setup_ood.log
```

脚本将官方源码放在 `runs/cache/LIBERO-PRO/`，评估文件放在 `runs/data/libero_pro/`，使用已有 `runs/env/simulator/`，无需第三个 Python 环境。原始 LIBERO 和 PRO 的源码、配置目录分开；每个条件都用独立 Python 进程运行，避免导入缓存混用后端。

下载完成报告是 `runs/reports/ood_assets.json`。下载脚本会记录每个文件的 SHA256；评估前再次核对固定代码版本、完整文件列表和哈希。报告 `assets_verified` 只代表资源完整，不代表模拟器已验收。

已有资源时，可单独检查，不重复下载：

```bash
runs/env/openpi/bin/python scripts/prepare_ood.py --verify-only
```

## 2. 启动策略服务

终端 A，先 `cd` 到仓库根目录，加载待评估组的真实训练 checkpoint：

```bash
source scripts/env.sh
runs/env/openpi/bin/python scripts/run.py serve spatial --repo-id local/pi05_3d_69 \
  --checkpoint runs/checkpoints/pi05_3d_spatial/paper_seed42/29999 --port 8000
```

30,000 steps 最后保存目录通常是 `29999`，以实际目录为准。baseline 对应替换 variant 和 checkpoint 路径。服务端现在发送 variant/checkpoint 元数据，客户端会拒绝组别不匹配或缺少元数据的旧服务；更新代码后要重启服务。

## 3. 首次先验收少量真实 OOD 回合

终端 B，进入相同仓库，等服务就绪后执行：

```bash
source scripts/env.sh
set -euo pipefail
for condition in task position; do
  for suite in libero_object libero_spatial; do
    runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite "$suite" \
      --condition "$condition" --trials 1 --max-tasks 1 --run-name preflight_spatial \
      2>&1 | tee "runs/logs/preflight_spatial_${suite}_${condition}.log"
  done
done
```

这只跑四个回合，结果标记 `partial_acceptance`，不能进入正式结果表。检查无异常、指令与目标正确、位移有限且合理之后再扩大评估。它不能代替对训练数据/几何的核对；部分失败或改用其他 checkpoint 重跑时，使用新的 `--run-name` 保存结果。预检结束后重启模型服务再做正式评估，避免预检消耗的模型随机数影响正式运行的起点。

## 4. 每组完整评估，以及两组汇总

同一个模型服务可以连续服务 ID、Task 和 Position。终端 B：

```bash
source scripts/env.sh
set -euo pipefail
bash scripts/evaluate_all.sh spatial 8000 paper_seed42 2>&1 | tee runs/logs/eval_spatial_all.log
```

`evaluate_all.sh` 按 ID、Task、Position 顺序分别运行 Object 和 Spatial，共六项。每项 10 任务 × 50 回合 = 500 回合；每组共 3,000 回合。也可以直接选单项：

```bash
runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite libero_object \
  --condition task --trials 50 --run-name paper_seed42
```

baseline 则启动 baseline checkpoint 的服务，执行 `bash scripts/evaluate_all.sh baseline 8000 paper_seed42`。两组运行期间不要修改评估代码/配置；保持相同 seed、初始状态索引和控制参数。

两组全部完成后生成成功率对照表：

```bash
runs/env/simulator/bin/python scripts/summarize_evaluation.py --run-name paper_seed42
```

汇总脚本要求两组十二项结果都完整、没有回合异常，且协议、评估代码提交号、seed、replan 相同、每组始终使用同一个 checkpoint，并逐回合核对任务、指令、目标及 BDDL/初始状态哈希。部分验收、异常或不配对的结果会被拒绝，不输出混合比较。

## 结果和失败处理

结果统一在 `runs/evaluation/paper_seed42/`：

- `{variant}_{suite}_{condition}.jsonl`：逐回合成功/错误、实际指令、抓取/放置目标、状态索引及文件哈希。
- 同名 `.metadata.json`：代码提交号、协议、官方资源 manifest 哈希、模型服务 checkpoint 和参数。
- 同名 `.summary.json`：成功率、错误数、评估规模和 `full_condition` / `partial_acceptance` 标记。
- `comparison.csv`、`comparison.json`：两组 ID/Task/Position 的 Object、Spatial、Avg，单位为百分数。

推理/仿真异常会记入 JSONL，最终命令失败；检查 summary 的 `errors` 为 0 后再解释成功率。中途失败不会自动续接部分 JSONL：保留原目录，修复后用新 run-name 重新评估两组，保证可汇总的十二项结果位于同一目录、来自同一代码版本。若只是某项尚未开始，且代码与配置都未变，可以用单项命令补齐它。已有结果不会自动覆盖。

本地单元测试只能覆盖任务选择、指令/目标切换、结果配对与资源完整性逻辑。真实权重、EGL、相机几何、仿真状态兼容性和完整 6,000 回合仍需算力机器验收。
