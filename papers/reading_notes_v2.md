# Direct Action-Head Injection：v2 精读与复现准备

阅读日期：2026-10-20。本文依据用户指定的 `arXiv-2606.27663v2` 全部正文、附录、实验表和图表源码整理。尚未开始复现、下载权重或训练。

## 1. 先锁定版本与证据范围

- 论文：*Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization*。
- 本次基准版本：本地 v2 LaTeX 源码；[arXiv v2](https://arxiv.org/abs/2606.27663v2) 标注 2026-10-01 修订。
- **同目录现有中文命名 PDF 是 v1**：提取首页可见 `arXiv:2606.27663v1 [cs.RO] 26 Jun 2026`，共有 22 页。不能用它替代 v2 核对真机结果或附录配置。
- 本次图像检查使用 v2 源码内自带的 PDF 图，已查看总体管线、8 个真机任务、失败案例和任务数量扩展图。
- [作者项目页](https://3dgnd-head-injection.github.io/)出现在 v2 摘要。本次网页工具无法读取，直接 HTTPS 请求也遇到 SSL 连接错误；**没有核实官方代码、权重和数据是否已公开**。不能把访问失败解释为尚未开源。
- 以下“论文明确”来自本地源码；“实现推导/建议”是根据公式整理的复现工作，不代表作者代码；“待核实”表示论文不足以唯一确定。

## 2. 论文到底解决什么

论文研究两个测试时变化：

1. **Position Perturbation**：指令不变，物体或放置区域移动，考察空间泛化。
2. **Task Perturbation**：场景布局保留，指令改为抓取/放置场景中的另一目标，考察是否服从当前指令，而非重放场景关联的训练轨迹。

“Task generalization”在这里限于抓放技能内部的新指令—场景配对、目标实例或目标区域变化。它不等于学会新技能，也没有证明开放世界任意操作能力。

核心命题是：**有可靠任务定位之后，将定位表达为夹爪到目标的三维位移，并通过动作头现有 AdaLN 通道输入，比把坐标写进文字或画到图像上更有效。**

完整系统仍依赖 RGB、语言、本体状态、外部定位、深度、相机标定和阶段切换。“一个点”指新增的稀疏几何条件，不是说机器人只看一个点就完成任务。

## 3. 方法逐步展开

### 3.1 输入、输出与数据依赖

一次策略决策至少涉及：

| 量 | 含义 | 来源/坐标系 |
|---|---|---|
| RGB 图像 | 第三人称和腕部视觉输入 | 原 VLA 输入 |
| 指令 | 当前抓放任务 | 原 VLA 输入，也用于外部定位 |
| 本体状态 | 末端姿态、夹爪等 | 原 VLA 输入 |
| `(u,v)` | 当前子目标的二维像素 | 仿真 oracle；真机 Qwen3-VL-4B |
| `D` | 与定位 RGB 对齐的公制深度图 | 第三人称 RGB-D |
| `K` | 相机内参 | 对应定位图像的坐标系 |
| `T_base_cam=(R,t)` | 相机坐标到机器人基座坐标的变换 | 外参标定 |
| `p_g` | 当前夹爪位置 | 机器人状态/正运动学，基座系 |
| action chunk | 后续多步动作 | 原 flow-matching 动作头 |

### 3.2 从二维像素到三维目标

先取像素深度 `z=D[v,u]`。源码公式用 `D(u,v)` 表示像素取值；实际数组一般按 `[v,u]` 索引，编码时必须确认。

\[
\mathbf p_t^{cam}=zK^{-1}[u,v,1]^T
=\left[\frac{(u-c_x)z}{f_x},\frac{(v-c_y)z}{f_y},z\right]^T
\]

\[
\mathbf p_t^{base}=R\mathbf p_t^{cam}+\mathbf t,
\qquad \mathbf d=\mathbf p_t^{base}-\mathbf p_g^{base}.
\]

关键含义：

- 注入的不是目标绝对坐标，而是 **3 维位移 `d`**。
- `d` 在机器人基座轴下表达，并非夹爪局部坐标向量。
- 反投影得到的是沿相机射线可见的表面点，不一定是物体质心、最优抓取点或期望 TCP 位姿。
- 这里的深度是针孔模型的光轴深度。若数据提供的是射线距离或仿真 depth buffer，需要转换。
- 两个位置必须先统一坐标系与单位，再相减。
- 相对位移消除了共同平移，但**不自动提供旋转不变性**；坐标轴变化会改变 `d`。

### 3.3 空间 MLP

论文附录明确给出：

```text
Linear(3, 1024) → ReLU → Linear(1024, 1024)
```

两种基座都使用 1024 维隐藏层和输出层。输出与所接入的动作头条件通道维度对应。

\[
\mathbf z_{spatial}=W_2\operatorname{ReLU}(W_1\mathbf d+b_1)+b_2.
\]

初始化：第一层标准初始化；**第二层权重和偏置都为零**。于是训练第 0 步 `z_spatial=0`，原时间条件保持不变。

实现推导：若两层 Linear 都含 bias，新增参数为

\[
(3\times1024+1024)+(1024\times1024+1024)=1,053,696.
\]

这是约 105 万参数，不是无参数方法；也不需要额外点云编码器。零输出初始化使首个反传步的第一层梯度为零，第二层更新后第一层开始获得梯度，这是该结构的正常行为。

### 3.4 注入位置

原动作头已有扩散/流时间条件 `z_time`。论文的新增路径是：

\[
\mathbf c=\mathbf z_{time}+\mathbf z_{spatial},
\qquad(\gamma_l,\beta_l)=\operatorname{Linear}_l(\mathbf c).
\]

各 DiT block 复用现有 AdaLN 条件投影，以其输出调制归一化特征。

必须保留的结构选择：

- 在 **AdaLN 条件投影前** 将空间嵌入加到时间嵌入上。
- 复用原动作头的投影；不额外为每层增加一套空间投影。
- 没有把空间向量加入 VLM 的语言 token 或图像 token。
- 没有把空间嵌入直接加到动作 token 上。
- 没有独立几何监督、目标位置回归损失或额外 grasp loss。

论文给的是抽象 `gamma,beta` 公式。实际基座代码可能还带残差 gate、不同归一化约定、多分支条件投影或不同命名。**必须按选定基座版本的计算图核实接入点，不能仅凭论文示意重写原 AdaLN。**

### 3.5 训练与推理

论文明确：固定外部定位源；空间 MLP 与 VLA 联合、端到端优化，使用原 flow-matching action loss。

“不改 backbone”是架构和新增输入路径不变，**不能推导成冻结整个 VLM，仅训练 MLP**。具体哪些模块可训练、哪些冻结、是否使用 LoRA，源码没有完整列出，仍需官方配置。

实现推导的接口示意如下，不是作者代码：

```python
# 相机/几何预处理，示意形状
p_t = lift_to_base(point_uv, depth, intrinsics, extrinsics)  # [B, 3]
d = p_t - gripper_position_base                            # [B, 3]
z_spatial = spatial_mlp(d)                                 # [B, 1024]

# 原动作头内部：融合后传给已有条件化路径
condition = time_embedding + z_spatial                     # 广播方式按基座实现
prediction = original_action_head(..., condition=condition)
loss = original_flow_matching_loss(prediction, ...)
```

该示意没有指定时间插值方向、速度场符号、张量额外维度和采样器；这些应保持基座原实现，不能把不同 flow-matching 约定混接。

## 4. 定位与抓放阶段切换

### 4.1 仿真定位：不只是“随便取中心”

默认点选择以附录为准：

- 对可获得显式 mask 的物体或放置容器，使用 **distance transform 最大值像素**，即离 mask 边界最远的内部像素。
- 该像素不一定等于 bounding-box 中心、mask 质心或物体 3D 质心。
- 对“盘子左边”等无显式 mask 的抽象区域：先取区域的仿真 3D 中心，投影到二维，再查该像素深度重新反投影。
- 最终输入的三维点必须来自可见像素的深度 lifting，不能直接拿仿真质心当 `p_t`。

这有一个需要准确表述的边界：**最终三维目标来自可见像素，并不表示系统完全不使用仿真特权信息**。目标 mask、抽象区域中心，以及阶段切换使用的物体高度，仍由仿真信息支持。

49 个附加任务的选择约束是目标/区域可获得 mask，或者抽象区域中心投影无遮挡。例如柜子第二层中心投影落在上层表面的任务被排除。完整名单位于原始附录表：

[49 个 LIBERO-90 任务](arXiv-2606.27663v2/sections/appendix/assets/tables/libero_90_tasks.tex)。

### 4.2 子目标切换

每次 episode 开始记录被抓物体的初始高度 `z_obj,0`。

```text
reach-to-grasp：当前目标 = 被抓物体
当 z_obj - z_obj,0 > 0.01 m：切换到 transport-to-place
transport-to-place：当前目标 = 放置物体或区域
```

阈值基于**物体升高**，不是夹爪升高，也不是夹爪闭合信号。

论文没有完整规定：阈值防抖、物体掉落后是否切回、遮挡时如何处理、chunk 中途切换如何同步。不得把自选状态机细节说成论文原设定。

### 4.3 真机定位

- 直接使用未微调的 **Qwen3-VL-4B**。
- 输入任务指令和前视 RGB，在一次查询中同时返回两条记录：`role=target` 和 `role=placement`。
- JSON 的 `point_2d` 按 `[0,1000]` 归一化，解析后缩放到输入图像分辨率，再按当前阶段选择对应点。
- `target` 是待抓物体；`placement` 是放置物体/区域。每次查询给出两点不等于每步都查询，论文未明确更新频率。

附录 prompt 的功能格式为：

```json
[
  {"point_2d": [x, y], "label": "object name", "role": "target"},
  {"point_2d": [x, y], "label": "region name", "role": "placement"}
]
```

逐字 prompt 保存在 [真机设置源码](arXiv-2606.27663v2/sections/appendix/real_world_experiment_setup.tex)。

**真机复现的明显缺口**：论文同样用“物体升高 1 cm”切换阶段，却未详细说明如何在真机持续估计这个物体高度。一次性的二维定位并不足以唯一确定这一过程，不能默认作者使用夹爪高度替代。

## 5. 仿真数据、训练与评估协议

### 5.1 数据范围

- 默认 69 个任务 = LIBERO-Object 10 + LIBERO-Spatial 10 + LIBERO-90 精选 49。
- 20 任务实验仅使用 Object + Spatial，用来检验数据规模影响。
- 不能使用任意 49 个 LIBERO-90 任务替代论文名单。
- 论文说明原轨迹相对单模态、初始状态随机化有限，69 任务扩展用于增加任务多样性。
- 源码没有给出完整训练数据清洗、每任务保留轨迹数、采样权重或转换代码。

### 5.2 主要仿真超参数

| 项目 | GR00T-N1.6 | π0.5 |
|---|---:|---:|
| action chunk 长度 | 16 | 10 |
| global batch size | 384 | 256 |
| 训练步数 | 20,000 | 30,000 |
| optimizer | AdamW | AdamW |
| peak LR | 1e-4 | 5e-5 |
| schedule | warmup 后 cosine decay | warmup 后恒定 |
| warmup | 5%，约 1,000 步 | 10,000 步 |
| weight decay | 1e-5 | 1e-10 |
| gradient clipping max norm | 1.0 | 1.0 |
| 输入分辨率 | 256 | 224×224，含 padding |

GR00T 视觉增强：最短边 resize 到 256；按最短边 0.95 裁成正方形后 resize 回 256；brightness 0.3、contrast 0.4、saturation 0.5、hue 0.08；关闭随机旋转。

π0.5 视觉增强：padding/resize 到 224×224；第三人称各空间维度 crop 0.95，再 bilinear resize，旋转 ±5°；腕部无几何增强；brightness 0.3、contrast 0.4、saturation 0.5；无 hue jitter。

其余超参数依赖作者所用官方 recipe，论文没有锁定所有版本。

### 5.3 评估规模

- Object、Spatial 两个 suite 各 10 个任务。
- 每任务 50 rollouts，因此每 suite、每种条件 500 次。
- 合并两 suite 后每种条件 1,000 次；ID、Task、Position 三类合计每个模型 3,000 次。
- 保留 suite 分项，不能只看总平均。尤其 GR00T + Ours 的 Position Spatial 仅 41.0%，明显低于其 Position Object 的 79.4%。
- 论文未给出完整训练随机种子、多次独立训练方差、checkpoint 选择准则和所有 rollout 控制细节。

## 6. 主结果与消融的准确读法

### 6.1 同数据、同基座的仿真对比

数值均为成功率百分数；Avg 为 Object/Spatial 平均。

| 模型 | ID Object | ID Spatial | ID Avg | Task Object | Task Spatial | Task Avg | Position Object | Position Spatial | Position Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π0.5 | 94.8 | 96.0 | 95.4 | 20.0 | 54.6 | 37.3 | 36.0 | 58.0 | 47.0 |
| π0.5 + Ours | 97.0 | 98.2 | 97.6 | 70.6 | 81.2 | 75.9 | 75.0 | 69.4 | 72.2 |
| GR00T-N1.6 | 93.8 | 93.2 | 93.5 | 10.0 | 52.4 | 31.2 | 29.2 | 27.0 | 28.1 |
| GR00T-N1.6 + Ours | 99.6 | 92.4 | 96.0 | 70.4 | 84.6 | 77.5 | 79.4 | 41.0 | 60.2 |

GR00T 的提升分别为 Task +46.3、Position +32.1 **个百分点**；π0.5 对应 +38.6、+25.2 个百分点。

ID 总均值提高不意味着每个分项都提高：GR00T 的 ID Spatial 从 93.2 降至 92.4。

外部模型参考：OpenVLA 的 ID/Task/Position 平均为 86.6/24.1/0.0；π0 为 97.8/32.2/5.9；MolmoAct 为 91.2/20.4/2.2。FALCON 在论文同 69 任务数据下重训后为 98.2/26.6/6.4。

这些参考模型不能一概视作与两组自家基座完全相同的训练控制变量。FALCON 失败的“稠密几何易过拟合”是作者解释，实验没有单独证明其因果机制。

### 6.2 2D/3D 表示与注入方式

所有增强变体从相同 oracle 二维点开始；3D 变体还通过深度和标定获得公制三维信息。不是所有变体拥有完全相同的模态输入。

| GR00T 变体 | 69 任务 Task | 69 任务 Position | 20 任务 Task | 20 任务 Position |
|---|---:|---:|---:|---:|
| baseline | 31.2 | 28.1 | 28.9 | 3.4 |
| 2D via Text Prompt | 42.5 | 31.7 | 32.1 | 15.3 |
| 2D via Visual Prompt | 40.3 | 30.4 | 29.6 | 1.7 |
| 2D via AdaLN | 47.7 | 37.8 | 39.6 | 19.4 |
| 3D via Text Prompt | 38.5 | 39.4 | 32.9 | 5.4 |
| 3D via AdaLN | 77.5 | 60.2 | 62.0 | 41.5 |

对照实现：

- 2D 文本：像素除以宽、高到 `[0,1]`；作者试过 `[0,1000]`，前者较好。抓取和放置阶段使用不同追加语句。
- 2D 视觉：目标位置叠加十字标记，文本追加“抓/放到十字标记位置”的说明。
- 2D AdaLN：输入四元组 `(x_t,y_t,x_t-x_g,y_t-y_g)`；`(x_g,y_g)` 为夹爪投影。源码未完整交代此变体的归一化细节。
- 3D 文本：追加的是基座系目标绝对坐标 `p_t`；Ours 注入的是相对位移 `d`。

因此这组结果有力支持论文组合设计，但不是完全正交的全因子实验。尤其 3D 文本与 3D AdaLN 还存在绝对坐标/相对位移的表示差异，不能把全部差距只归因于 token 化。

20 任务 Ours 在 Task 62.0、Position 41.5 上均高于其他 69 任务变体。这支持本设置下的结构收益，不代表增加数据普遍无用；Ours 从 20 到 69 任务也继续提高。

### 6.3 空间输入与注入通道消融

这里的 Avg 同时平均两 suite 和两种 OOD 条件，共四格，不是只平均 Task 或 Position。

| 空间输入 | OOD Avg |
|---|---:|
| baseline | 29.7 |
| 只有 `p_t` | 34.3 |
| `p_t,p_g` | 53.7 |
| 只有 `d` | 68.9 |
| `p_t,p_g,d` | 59.7 |

| 注入方式 | OOD Avg |
|---|---:|
| 通过 robot state 拼接 | 51.7 |
| In-Context，额外 spatial token | 58.5 |
| AdaLN | 68.9 |

关键证据是“显式相对关系 + 直接条件化”优于给更多绝对坐标或仅增加 token。作者将绝对位置较差归因为不同场景桌面高度等变化；这应视为与消融一致的解释，而非被单独识别的唯一原因。

### 6.4 深度噪声

训练和测试分别加入标准差 0、0.01 m、0.03 m 的高斯深度噪声。下表同样为四格 OOD Avg。

| 训练深度 | 测试 clean | 测试 σ=0.01 m | 测试 σ=0.03 m |
|---|---:|---:|---:|
| clean | 68.9 | 68.4 | 70.2 |
| σ=0.01 m | 61.6 | 61.3 | 61.5 |
| σ=0.03 m | 57.9 | 59.3 | 57.2 |

可以说测试噪声下表现接近、训练噪声降低最终水平。不能从 70.2 高于 68.9 推断噪声显著提升效果，因为没有训练种子方差或显著性分析。

加性高斯深度噪声没有覆盖标定系统偏差、深度空洞、错物体深度、透明反光物体、时间不同步等误差类型。

### 6.5 定位噪声

测试时二维点噪声标准差按目标 bounding-box 尺寸的比例设定；**扰动后若点不在正确物体 mask 内，重采样直到落在 mask 内**。

| 定位条件 | OOD Avg |
|---|---:|
| 无噪声 | 68.9 |
| σ=10% | 67.3 |
| σ=30% | 62.1 |
| σ=50% | 59.9 |
| mask 内随机点 | 59.3 |

所以它测的是正确目标内部的点位变化。不能据此声称模型能承受 50% 概率认错目标，或任意背景点噪声。

## 7. 真机设置与结果

### 7.1 硬件、任务与数据

- Franka Emika Panda。
- 两台 RealSense D435：前视提供 RGB-D，腕部提供 RGB。
- 4 种布局，每布局 2 个任务；共 8 任务，每任务 40 演示，共 320 演示。
- 演示采集有适度位置变化，但保持整体布局。
- 每任务、每条件 10 rollouts；每条件 80 次，每模型三类条件合计 240 次。
- OOD 的物体和区域在其他训练任务中被操作过；新颖性在配对或位置，不是全新物体和技能。

从 v2 真机任务图读取的 ID 任务：

| Task | 抓取对象 | 放置目标 |
|---|---|---|
| 1 | 蓝色方块 | plate |
| 2 | 红色方块 | bowl |
| 3 | 橙子 | plate |
| 4 | 柠檬 | bowl |
| 5 | 蓝色方块 | pink bowl |
| 6 | 橙子 | green bowl |
| 7 | 红色方块 | pink plate |
| 8 | 柠檬 | green plate |

Task 1–2 的容器在图中为绿色；Task 3–4 为粉色。具体 OOD 布局以原图为准，图中箭头还区分了改抓取对象与改放置区域两类 task 扰动。

### 7.2 状态和动作

- 状态 8 维：TCP 基座系位置 3 + rotvec 3 + 两指实测宽度 2。
- 动作 7 维：末端位置/rotvec 命令 6 + 绝对夹爪宽度命令 1。
- 末端动作采用相对于当前末端位姿的变换，旋转在 SO(3) 上复合。

**不能直接将两个 rotvec 做减法当相对旋转**。相对变换左/右乘、平移在哪个参考系、action chunk 各步相对于哪一时刻，需要通过实际数据与基座转换代码确认，源码描述没有消除所有约定歧义。

### 7.3 真机训练

| 项目 | GR00T-N1.6 | π0.5 |
|---|---:|---:|
| chunk | 24 | 24 |
| global batch | 256 | 256 |
| steps | 20,000 | 20,000 |
| optimizer | AdamW | AdamW |
| LR | 1e-4 | 5e-5 |
| schedule | warmup 后 cosine | warmup 后恒定 |
| warmup | 1,000 左右 | 5,000 |
| weight decay | 1e-5 | 1e-10 |
| gradient clipping | 1.0 | 1.0 |

GR00T 真机图像先在短边 zero padding 为正方形，再 resize、各边 crop 到 0.95、resize 回 256×256；颜色增强与仿真一致，无随机旋转。π0.5 真机的增强方式与其仿真设置相同。

### 7.4 真机成功率

| 模型 | ID | Task | Position |
|---|---:|---:|---:|
| π0.5 | 85.0 | 32.5 | 27.5 |
| π0.5 + Ours | 91.3 | 78.8 | 61.3 |
| GR00T-N1.6 | 68.8 | 3.8 | 3.8 |
| GR00T-N1.6 + Ours | 92.5 | 63.8 | 66.3 |
| GR00T + 2D Text | 87.5 | 5.0 | 7.5 |
| GR00T + 2D Visual | 88.8 | 6.3 | 5.0 |
| GR00T + 2D AdaLN | 87.5 | 18.8 | 16.3 |
| GR00T + 3D Text | 86.3 | 5.0 | 5.0 |

真机确实使用了 VLM 定位和传感器深度，说明组合设计可在该设置落地。但每任务只有 10 次，平均值不能掩盖失败任务，也没有给出足以确定效果方差的多次训练统计。

### 7.5 失败模式

GR00T + Ours 仍有两个 0/10：

- Task 3 的 Task Perturbation：应把橙子放进碗里，但提前放到训练时使用的盘子上。作者推测训练中盘子的较大空间范围，与邻近碗的定位区域发生混淆。
- Task 5 的 Position Perturbation：尝试朝移动后的放置区调整，但最终仍失败。

作者观察这些轨迹里的二维点几乎都正确，因此这些失败不能简单归为定位错误。模型在正确目标条件下仍可能受训练轨迹、局部几何和动作执行影响。

## 8. 后续复现必须核实的缺口

| 项目 | 论文已知 | 仍需核实 |
|---|---|---|
| 基座版本 | GR00T-N1.6、π0.5 | checkpoint 名称、代码 commit、processor、依赖锁定 |
| 可训练参数 | MLP 与 VLA 联合训练 | VLM/视觉编码器/动作头各自冻结设置，LoRA 与否 |
| AdaLN 接入 | 与时间嵌入相加、复用投影 | 两基座具体文件、条件通道维度/分支、gate、dtype/broadcast |
| 几何量尺度 | 公制坐标、相对位移 | 是否额外标准化、裁剪或数据统计归一化 |
| 训练数据 | 69 任务及 49 项名单 | 轨迹保留数、转换、采样权重、有效帧筛选 |
| 逐帧几何 | 原理明确 | 已有数据是否含 depth/mask/相机参数，缺失时如何确定性重放 |
| 子目标状态 | 物体升高 >1 cm | 离线标注、推理频率、防抖/回退、真实物体高度获取 |
| Qwen 定位 | 4B、一次返回两点 | 精确 checkpoint、解码参数、刷新/跟踪频率、JSON 异常与遮挡处理 |
| 图像增强 | crop/resize/rotation 数值 | 像素、深度、K、marker 的一致变换与计算时机 |
| action chunk | 仿真 16/10，真机 24 | 每次实际执行几步、重规划频率、阶段切换同步 |
| 评估 | 每任务 50/10 次 | LIBERO-PRO commit、perturbation 配置、seed、horizon、成功判据 |
| 训练资源 | batch 与 steps | GPU 型号/数量、精度、显存、训练时间与推理吞吐 |
| 模型选择 | 未完整报告 | 最终/最佳 checkpoint、验证集、重复训练统计 |

## 9. 复现推进顺序（建议，尚未执行）

### 第一步：固定目标与建立原基座基线

先选择一种基座，锁定环境、checkpoint 和官方训练配置。20 任务适合验证机制；若目标是对齐主表，最终必须回到相同 69 任务训练与完整评估。

保持 RGB、状态、动作、归一化与评估协议一致，先确认无新增模块的 ID 和 OOD 表现。单纯跑通训练 loss 不算基线复现成功。

### 第二步：独立验收几何与阶段标签

在不训练策略前验证：

1. 像素反投影再投影能回到原像素；外参方向和坐标轴正确。
2. 原始深度/仿真 depth buffer 已转换成米制光轴深度。
3. `p_t`、`p_g` 在同一基座系；位移方向确为 target minus gripper。
4. 掩膜点符合 EDT 最大值选择；抽象区域执行“投影—再 lifting”。
5. 在多条 demo 上画出目标点、`d` 和阶段切换时刻，确认升高阈值对应物体。
6. 增强前后像素、深度和内参一致。可选择在原始几何流先算 `d`，但必须记录此实现选择，并核对作者方案。

如果原始演示没有 depth、segmentation 或相机参数，需要先确定能否从保存的仿真状态重放恢复；不能默默用质心坐标替代论文输入。

### 第三步：验收最小模型改动

增加零末层初始化的 3→1024→1024 MLP，在现有 AdaLN 条件路径融合。

值得运行的实质性检查：

- 在相同原权重、相同输入、相同 noise/time、eval 模式下，新增模块初始化后的动作头输出与原模型一致（按数值精度设容差）。
- 确认新参数进入 optimizer，反向梯度和 checkpoint 保存/重载正常。
- 测试 batch/chunk/time 维度，不依赖偶然 broadcasting。
- 禁用定位分支或输入零 embedding 时可回到原条件路径。

### 第四步：先用 oracle 验证机制，再验证外部定位

在相同训练预算下比较 baseline 和 3D AdaLN，分别记录 ID/Task/Position 的 Object、Spatial 分项；可补 2D AdaLN 对照，以区分“多了一条条件通道”与三维几何本身的收益。

机制成立后再扩展到 69 任务、完整表格及真机。引入 VLM 时，单独统计目标识别、有效深度、阶段切换和策略执行的失败比例，避免所有失败混成一个成功率。

## 10. 源码索引

下列路径均相对于本笔记所在目录：

| 需要查什么 | 原始来源 |
|---|---|
| 论文入口、作者、所有 input | [main.tex](arXiv-2606.27663v2/main.tex) |
| 核心方法与联合训练 | [method.tex](arXiv-2606.27663v2/sections/method.tex) |
| lifting、1024 维 MLP、零初始化、变体 prompt | [implementation_details.tex](arXiv-2606.27663v2/sections/appendix/implementation_details.tex) |
| 默认任务数量与阶段切换、实验论证 | [simulation_exps.tex](arXiv-2606.27663v2/sections/simulation_exps.tex) |
| EDT、抽象区域、仿真超参数 | [simulation_experimental_setup.tex](arXiv-2606.27663v2/sections/appendix/simulation_experimental_setup.tex) |
| 仿真主表 | [main table](arXiv-2606.27663v2/assets/tables/main.tex) |
| 69 任务对照 | [controlled_experiment.tex](arXiv-2606.27663v2/assets/tables/controlled_experiment.tex) |
| 空间输入/注入通道消融 | [ablation.tex](arXiv-2606.27663v2/assets/tables/ablation.tex) |
| 20 任务对照 | [controlled_experiment_20_task.tex](arXiv-2606.27663v2/sections/appendix/assets/tables/controlled_experiment_20_task.tex) |
| 深度与定位噪声完整分项 | [depth table](arXiv-2606.27663v2/sections/appendix/assets/tables/depth_noise_experient_table.tex)、[grounding table](arXiv-2606.27663v2/sections/appendix/assets/tables/grounding_noise_experiment_table.tex) |
| Qwen prompt、状态动作、真机超参数 | [real_world_experiment_setup.tex](arXiv-2606.27663v2/sections/appendix/real_world_experiment_setup.tex) |
| 真机汇总/每任务结果 | [real summary](arXiv-2606.27663v2/assets/tables/real_world_experiment.tex)、[per-task results](arXiv-2606.27663v2/sections/appendix/assets/tables/real_world_experiment_table.tex) |
| 两种 0/10 失败分析 | [qualitative_results.tex](arXiv-2606.27663v2/sections/appendix/qualitative_results.tex) |
| 作者自述边界 | [discussion_and_limitations.tex](arXiv-2606.27663v2/sections/discussion_and_limitations.tex) |
| v2 原始管线图 | [overall_pipeline_fig.pdf](arXiv-2606.27663v2/assets/figures/overall_pipeline_fig.pdf) |
| v2 原始真机任务图 | [real_world_task_fig.pdf](arXiv-2606.27663v2/sections/appendix/assets/figures/real_world_task_fig.pdf) |

后续继续本项目时，应优先读取本笔记和对应原始附录；任何代码层面的确定结论，仍须对照实际采用的基座代码和官方实现验证。
