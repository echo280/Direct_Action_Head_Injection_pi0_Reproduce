# 与论文对齐的内容和本版实现选择

## 已按论文实现

- 2D mask 的 EDT 内部最远点，经公制深度和相机参数 lifting。
- 相对位移 `target_base - gripper_base`，空间分支没有额外标准化。
- ReLU 两层 MLP，完整 π0.5 的隐藏和输出维度为 1024，末层全零。
- 融合在 π0.5 的时间条件 MLP 输出之后、现有 adaRMSNorm 条件投影之前。
- 原 flow-matching loss；两组相同 pi05_base 初始化、数据和主要训练预算。
- 物体升高超过 1 cm 后从抓取点切换到放置点。
- 精确的 69 任务名单和 20 任务名单。

## 需要远端验收

- 官方权重恢复时新 MLP 参数保留初始化，原模型权重严格匹配。
- 数据转换前后 episode/frame 身份；sidecar 使用元数据指纹及每帧 ID 防止错位。
- RGB/深度/segmentation 方向与相机内外参；仿真 world 到 robot base 的转换。
- 原 HDF5 `states[i]` 对应 `actions[i]` 的时间约定，和原始 RGB 记录的一致性。
- 抓取遮挡/空 mask、抽象区域遮挡以及投影可见性的处理。
- 小模型通过后，再跑真实 pi05_base 的少量训练步、保存/重载和推理检查。

## 本版工程选择，不能说成作者原设定

- 重放从状态重新渲染，保留全部帧，未使用其他 LIBERO 数据的 no-op 过滤。
- 仿真控制频率与数据 fps 设为 20，论文没有给出完整 fps 和清洗 recipe。
- 阶段切换锁存，物体掉落不自动回退。
- 原始 cv 几何流先计算公制位移，再独立做 VLA RGB 几何增强。
- 使用固定的一套 robot base 定义；公开源码没有作者相机/机器人转换代码可逐行核对。
- 保留官方默认 EMA 0.999，默认全量参数微调；论文冻结策略未完整报告。
- 数据加载 workers=0 便于首次排错，与性能优化无关；后续调整应两组一致。
- ID rollout 默认 replan 5 步，阶段变化时清空旧 chunk；论文没有给出这些控制细节。
- Task/Position 使用固定版本官方 LIBERO-PRO 的 `_task` / `_swap` suite 和配套预生成状态；作者未公开精确 benchmark commit、资源 revision 及全部 rollout 参数，本版显式记录于 `configs/evaluation.json`，仍需远端验证。

## 尚未完成

- LIBERO-PRO Task/Position 的真实仿真验收与完整评估结果（代码入口已接入）。
- 真机 Qwen 定位、真实物体高度跟踪、机器人控制。
- 论文的全部消融、深度/定位噪声实验、多 seed 统计。

本版用于让算力提供方建立环境并开始工程验收。达到论文数值之前，必须先完成上述数据与评估协议核对。
