# 评估协议

## 已提供的 ID 入口

先启动训练 checkpoint 的策略服务，在另一个终端运行模拟器客户端：

```bash
source scripts/env.sh
runs/env/openpi/bin/python scripts/run.py serve spatial --repo-id local/pi05_3d_69 \
  --checkpoint runs/checkpoints/pi05_3d_spatial/paper_seed42/29999 --port 8000
```

最后一次保存目录使用官方训练脚本的循环步编号，30,000 steps 时通常是 `29999`；以实际目录为准。

```bash
runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite libero_object --trials 50
runs/env/simulator/bin/python scripts/evaluate.py --variant spatial --suite libero_spatial --trials 50
```

baseline 要启动对应 baseline checkpoint 的服务，客户端 `--variant baseline`。
输出逐回合 JSONL 和汇总 JSON 到 `runs/evaluation/`；包含模拟器/推理异常的回合会记录错误并让最终命令失败。
重复实验要单独保留结果目录，不自动覆盖已有结果。

## OOD 待完成

官方 LIBERO-PRO：https://github.com/Zxy-MLlab/LIBERO-PRO
论文：https://arxiv.org/abs/2510.03827

本版没有选择其默认扰动配置冒充目标论文协议。Task 和 Position 适配尚未实现，`--condition task` 或 `position` 会明确报错。
需要先锁定 benchmark commit、扰动任务、初始化状态、horizon、目标重标注和 grounding 获得方式，然后接入同一 `spatial_displacement` 推理接口。

论文每 suite 10 任务、每任务 50 回合，Task/Position 分开统计。结果表应至少保留 Object/Spatial 分项，而不是只给总平均。
**ID 成功、小模型 loss 下降或脚本退出 0 都不能替代 OOD 评估。**
