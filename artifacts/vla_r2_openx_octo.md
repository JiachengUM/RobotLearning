# VLA 发展史调研笔记（第二环）：跨机构数据与开源生态 —— Open X-Embodiment / RT-X 与 Octo

> 定位：本笔记是 VLA 发展史主线（RT 系列 → 开源 VLA → π 系列）的第二环，回答"Google DeepMind 闭源 RT-1/RT-2 之后，如何通过跨机构数据联盟与开源权重把能力开放给社区"。
> 写作日期：2026-09-29。所有技术事实均附信源；无法从一手信源核实者标注【未查证】。

---

## 一、来源清单（Source List）

一手信源（论文 / 官方项目页 / 官方博客）：

1. Open X-Embodiment 论文摘要页（arXiv:2310.08864，2023-10-13 提交，v9 2025-05-14）：https://arxiv.org/abs/2310.08864
2. Open X-Embodiment 论文 PDF（v3，数据集规模描述）：https://arxiv.org/pdf/2310.08864v3
3. Open X-Embodiment 论文 PDF（v7，RT-1 架构细节）：https://arxiv.org/pdf/2310.08864v7
4. RT-X 官方项目页（数据集概览、RT-1-X/RT-2-X 结果）：https://robotics-transformer-x.github.io/
5. Google DeepMind 官方博客 "Scaling up learning across many different robot types"（2023-10-03）：https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types
6. RT-2 官方项目页（VLA / action-as-text-token 定义）：https://robotics-transformer.github.io/
7. Octo 论文摘要页（arXiv:2405.12213，2024-05-20 提交，RSS 2024）：https://arxiv.org/abs/2405.12213
8. Octo 官方项目页（模型规格、零样本/微调结果表）：https://octo-models.github.io/
9. Octo 论文 HTML 正文（v2，架构/训练/消融/局限）：https://arxiv.org/html/2405.12213v2
10. OpenVLA 论文摘要页（arXiv:2406.09246，2024-06-13，作为后续影响的直接证据）：https://arxiv.org/abs/2406.09246
11. π0 论文 PDF（跨 embodiment 训练方法学继承）：https://isaacbs.com/papers/pi0.pdf

佐证 / 二手信源（用于交叉核对数据集口径与后续采用情况）：

12. 将 RT-1-X 移植到 SCARA 机器人的论文（确认 OXE 发布时口径与 RLDS 格式）：https://arxiv.org/pdf/2409.03299
13. 视频学习综述中的 OXE 数据表（"1M+ trajectories across 527 skills"）：https://arxiv.org/html/2402.07127v3
14. OXE-AugE（2025，OXE 的大规模增强扩展，提及在 OpenVLA 与 π0 上微调）：https://arxiv.org/html/2512.13100

---

## 二、对象一：Open X-Embodiment 数据集 与 RT-X 系列（RT-1-X / RT-2-X）

- 论文：*Open X-Embodiment: Robotic Learning Datasets and RT-X Models*，Open X-Embodiment Collaboration（作者名单 190+ 人），arXiv:2310.08864，2023-10-13 提交，后发表于 ICRA 2024。["https://arxiv.org/abs/2310.08864"]
- 配套开放资源：数据集 + RT-X 预训练 checkpoint，托管于 robotics-transformer-x.github.io。["https://robotics-transformer-x.github.io/"]

### 2.1 它解决了什么问题

- **单机构数据不足**：机器人学习传统上"为每个应用、每个机器人、甚至每个环境单独训练一个模型"，而大规模多样化数据是 NLP/CV 出现通用基座模型的前提；机器人领域此前没有这种共享数据集。["https://arxiv.org/abs/2310.08864","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"]
- **跨 embodiment 泛化缺失**：不同机器人（单臂、双臂、四足）动作空间、传感器配置不同，此前无法用同一策略跨平台迁移；该工作要验证"一个 X-robot 策略能否借其他平台的经验提升本机能力"。["https://arxiv.org/abs/2310.08864"]
- **开源可复现性缺失**：Google 此前的 RT-1/RT-2 训练数据与多为内部/封闭，社区既无统一格式数据、也无可用预训练权重；本工作把数据归一化为统一格式并放出 checkpoint。["https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types","https://arxiv.org/pdf/2409.03299"]

### 2.2 关键技术

**(a) 数据归一化与混合（data mixture）**

- 把全球 34 个机器人研究实验室已有的 60 个机器人数据集汇聚、转换为统一格式（RLDS，Reinforcement Learning Datasets 格式），得到 1M+ 真实机器人轨迹。["https://arxiv.org/pdf/2310.08864v3","https://arxiv.org/pdf/2409.03299"]
- 动作被统一表示为**相对于机器人夹爪坐标系（gripper frame）的 7 维向量**：x、y、z、roll、pitch、yaw、夹爪开合（或这些量的速率形式）；对不涉及某些自由度的机器人，训练时把对应维置零。["https://robotics-transformer-x.github.io/"]
- 这一"夹爪系 7-DoF 归一化动作"是跨 embodiment 共训练能成立的关键接口。【已查证事实，来源为项目页】

**(b) RT-1-X：以 RT-1 为基座的共训练适配**

- RT-1 本身是 35M 参数、基于 Transformer 的机器人控制网络：输入 15 帧图像历史 + 自然语言，图像经 ImageNet 预训练 EfficientNet 处理，语言经 USE 嵌入，视觉与语言表征拼接后送入 Transformer。["https://arxiv.org/pdf/2310.08864v7"]
- RT-1-X 即"在上述机器人数据混合体上训练的 RT-1"，**网络架构与 RT-1 完全相同**；论文明确把性能提升归因于跨 embodiment 共训练数据而非架构变化。["https://robotics-transformer-x.github.io/","https://fileserver-az.core.ac.uk/download/618502492.pdf"]
- 结果：在小数据域，RT-1-X 比"各实验室在自己数据上训练的原始方法（Original Method）"或单数据 RT-1 平均高约 **50%**；在 6 个学术实验室的真实机器人上评测。["https://robotics-transformer-x.github.io/","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"]

**(c) RT-2-X：以 RT-2（VLA）为基座的共训练适配**

- RT-2 是把动作表达为**自然语言 token**、与文本一起训练的视觉-语言-动作（VLA）模型。["https://robotics-transformer.github.io/"]
- RT-2-X 即"在机器人数据混合体上共微调的 RT-2"，规模约 **55B 参数**，是当时在学术实验室执行未见任务的最大模型之一。["https://robotics-transformer-x.github.io/"]
- 结果：在 emergent skills（评测任务的物体/技能出现在其他机器人的数据集中、但不在 RT-2 自身数据里）上，RT-2-X 成功率约为 RT-2 的 **3 倍**；并能由介词细微变化（"on" vs "near"）调制底层动作，表现出空间关系理解。["https://robotics-transformer-x.github.io/","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"]

### 2.3 范式遗产（对 OpenVLA / π 系列的直接影响）

- **确立了"跨 embodiment 共训练"这一标准范式**：用一个高容量模型在异构机器人数据混合体上训练，能对多个机器人产生正迁移（positive transfer）。["https://arxiv.org/abs/2310.08864"]
- **成为社区事实标准数据集**：后续开源 VLA OpenVLA（7B，Llama 2 + DINOv2/SigLIP）即用 970k 条真实机器人演示训练，其代码库"内置在 Open X-Embodiment 数据上大规模训练 VLA 的支持"；OpenVLA 以约 1/7 的参数量、在 29 个任务上比 RT-2-X(55B) 绝对成功率高 16.5%——这正是站在 OXE 数据与 RT-2-X 基线之上的迭代。["https://arxiv.org/abs/2406.09246"]
- **为 π 系列提供方法学先验**：π0 论文明确采用 cross-embodiment training（把多种机器人构型、不同动作表示的数据汇入同一模型），该路线直接承自 RT-X 一线工作。["https://isaacbs.com/papers/pi0.pdf"]
- **开源权重基线**：RT-1-X 检查点可直接推理/微调，成为后来一系列工作（如把 RT-1-X 移植到 SCARA 臂）的起点。["https://arxiv.org/pdf/2409.03299"]

### 2.4 遗留问题

- **跨 embodiment 泛化有上限**：正迁移主要在"小数据域/同构桌面操作"上显著；对传感器构型、形态差异极大的平台（如四足导航、双臂移动操作）泛化并未被证明。【基于论文评测范围的归纳；评测集中在桌面操作】
- **RT-2-X 的动作粒度/分辨率问题**：RT-2/RT-2-X 把动作离散化为文本 token 输出，这种离散化接口天然受 token 分辨率限制，难以表达高分辨率、连续的精细动作；这是后续工作转向连续动作头（diffusion / flow-matching）的动机之一。["https://robotics-transformer.github.io/"]（具体分箱数【未查证】，此处不写死数字）
- **数据口径随时间漂移**：数据集发布后持续新增数据集，"1M+/60/34"是发布时口径，后续规模已增长（见数据表）。["https://arxiv.org/pdf/2409.03299"]

---

## 三、对象二：Octo（2023 预印 / 2024 发表，Berkeley–Stanford–CMU–Google DeepMind）

- 论文：*Octo: An Open-Source Generalist Robot Policy*，Octo Model Team，arXiv:2405.12213，2024-05-20 提交；发表于 RSS 2024（代尔夫特）。["https://arxiv.org/abs/2405.12213"]
- 机构：UC Berkeley、Stanford、Carnegie Mellon、Google DeepMind。["https://octo-models.github.io/"]
- 注：BibTeX 中年份写 2023，但 arXiv 提交为 2024-05，RSS 2024 正式发表。["https://octo-models.github.io/","https://arxiv.org/abs/2405.12213"]

### 3.1 它解决了什么问题

- **开源可复现的通用策略基线缺失**：在 Octo 之前，通用机器人策略要么闭源（RT-2-X 55B），要么不能灵活适配新传感器/动作空间；Octo 自称是"第一个能有效微调到新观测与动作空间、且训练管线+权重+数据全开源的通用机器人操作策略"。["https://arxiv.org/html/2405.12213v2"]
- **异构接口适配难**：不同机器人相机配置（有无腕相机）、动作空间（末端 delta vs 关节位置）、标注（有无语言）差异大；Octo 要让同一预训练模型不重新初始化大部分参数即可接入新接口。["https://arxiv.org/html/2405.12213v2"]
- **微调门槛高**：希望在消费级 GPU 上数小时内微调到新机器人，降低通用策略的使用门槛。["https://arxiv.org/abs/2405.12213"]

### 3.2 关键技术

**(a) 架构：transformer-first + 模块化 block-wise attention**

- 三段式：输入 tokenizer（语言/图像/观测）→ Transformer 主干 → readout head 出动作。["https://arxiv.org/html/2405.12213v2"]
- **语言**：冻结的 t5-base（111M），输出 16 个语言嵌入 token；实验发现更大编码器或微调编码器都无收益（归因于数据缺乏丰富自由文本标注）。["https://arxiv.org/html/2405.12213v2"]
- **图像**：不用大型 ResNet，而用**浅层 CNN patch 编码器**把图像切成 16×16 patch（第三人称相机 256 token，腕相机 64 token），把参数/算力集中在 Transformer 主干（"transformer-first"，类 ViT）。["https://arxiv.org/html/2405.12213v2"]
- **block-wise 掩码注意力**：观测 token 只能因果地关注同/更早时间步的观测 token 与任务 token；缺失模态（如无语言的数据）整体掩码。这种模块化设计使得微调时可直接增删观测/任务 token 而不动已预训练参数。["https://arxiv.org/html/2405.12213v2"]
- **readout token**：插入可学习 readout token（类似 BERT 的 [CLS]），被动读取前面的观测/任务嵌入，再由轻量动作头解码。["https://arxiv.org/html/2405.12213v2"]

**(b) 动作表示与解码：连续动作 + diffusion head + action chunking**

- 用**条件扩散解码头**预测连续、多模态动作分布；主干每次动作预测只前向一次，多步去噪全部在小扩散头内完成。["https://arxiv.org/html/2405.12213v2"]
- 扩散头为 3 层 MLP（隐维 256，残差 + LayerNorm），DDPM 目标、cosine 噪声 schedule、20 步去噪；消融显示该扩散头优于 MSE 头或离散化动作分布。["https://arxiv.org/html/2405.12213v2"]
- **相对动作表示**：训练混合中所有机器人统一用**末端执行器 delta（增量）控制动作**。["https://arxiv.org/html/2405.12213v2"]
- **action chunking**：一次预测未来一段连续动作（chunk），配合 receding horizon control；实验未发现 temporal ensembling 额外收益。["https://arxiv.org/html/2405.12213v2"]
- 观测历史仅用 **2 帧**（再多一帧收益显著递减）；goal image 用 hindsight goal relabeling。["https://arxiv.org/html/2405.12213v2"]

**(c) 模型规模与训练数据**

- 三档：Octo-Tiny 10M / Octo-Small 27M（12 层，hidden 384，6 头）/ Octo-Base 93M（12 层，hidden 768，12 头）；零样本性能随规模上升。["https://octo-models.github.io/","https://arxiv.org/html/2405.12213v2"]
- 预训练数据：从 Open X-Embodiment 中选 **25 个数据集、约 80 万条（800k）轨迹/episode**，覆盖不同 embodiment、场景、任务，且传感器（有无腕相机）与标签（有无语言）均异构。["https://octo-models.github.io/","https://arxiv.org/abs/2405.12213"]
- 训练：AdamW、batch 2048、300k 步，TPU v4-128 上约 14 小时；在单张 A5000(24GB) 上微调约 5 小时。["https://arxiv.org/html/2405.12213v2"]

**(d) 结果（项目页数据表）**

- 零样本（语言指令）成功率：WidowX 0.50 / UR5 0.70 / RT-1 Robot 0.80；对比 RT-1-X（0.20/0.35/0.60）显著更强，与 55B 的 RT-2-X（0.50/—/0.85）相当。["https://octo-models.github.io/"]
- 微调（约 100 条目标域演示，统一 recipe）：6 个评测设置平均 0.72，比次优基线高 52%；并能接入新观测（力/力矩）与新动作空间（关节位置控制）。["https://octo-models.github.io/"]
- 用 goal image 指定任务时，WidowX 上平均再高约 25%。["https://octo-models.github.io/"]

### 3.3 范式遗产

- **开源通用策略基线**：在 RT-1-X/RT-2-X 之后，Octo 把"全开源（代码+权重+数据）+ 可消费级 GPU 微调"做成默认配置，成为 OpenVLA 之前社区使用最广的开源通用策略之一；后续工作（如 NORA、各类 generalist policy 微调研究）直接以 Octo 为对比基线。["https://octo-models.github.io/","https://arxiv.org/html/2501.04693v3"]
- **连续动作头路线**：证明在 OXE 这种多模态演示数据上，diffusion 连续动作头优于离散 token 动作；这为后续 OpenVLA 之后的连续动作 VLA、乃至 π0 的 flow-matching 动作专家提供了参照（注意：Octo 本身无互联网 VLM 预训练，动作头是独立扩散头）。["https://arxiv.org/html/2405.12213v2"]
- **模块化可扩展设计**："增删 token 而不重初始化主干"的 block-wise attention + adapter 思路，成为后续跨 embodiment 适配研究的模板。["https://arxiv.org/html/2405.12213v2"]

### 3.4 遗留问题 / 能力边界

- **腕相机信息处理不佳**：论文自述"当前 Octo 难以充分利用腕相机信息，很多时候只用第三人称相机反而比第三人称+腕相机更好"。["https://arxiv.org/html/2405.12213v2"]
- **规模与语义 grounding 有限**：仅 93M 级、语言用冻结 t5-base，缺乏互联网视觉-语言预训练，开放词汇语义推理弱于 RT-2-X/OpenVLA 这类带大 VLM 的模型；这正是 OpenVLA(7B, Llama2+DINOv2+SigLIP) 后来居上的原因。["https://arxiv.org/html/2405.12213v2","https://arxiv.org/abs/2406.09246"]
- **预训练数据只是 OXE 子集**（25/60 数据集、800k/1M+），且评测集中在桌面操作；对长程、移动操作、接触富集任务的泛化有限。["https://octo-models.github.io/"]

---

## 四、Open X-Embodiment 数据集数据表（每个数字 + 来源）

> 注意口径：论文摘要、项目页、DeepMind 博客三处对"机构数/任务数"的表述存在**口径差异**（"21 institutions" vs "33 academic labs" vs "34 labs"），原因是统计口径（机构 vs 实验室）与发布后持续新增数据；下表逐条标注来源，不做强行统一。

| 指标 | 数值 | 来源口径 | 信源 |
|---|---|---|---|
| 真实机器人轨迹数 | 1M+（"more than 1 million episodes"） | 项目页 / 博客 | ["https://arxiv.org/pdf/2310.08864v3","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |
| robot embodiment 数 | 22（从单臂到双臂、四足） | 项目页 / 论文 | ["https://arxiv.org/pdf/2310.08864v3","https://robotics-transformer-x.github.io/"] |
| 汇聚的已有数据集数 | 60 个 | 项目页 / SCARA 移植文 | ["https://arxiv.org/pdf/2310.08864v3","https://arxiv.org/pdf/2409.03299"] |
| 参与实验室数 | 34 个 robotic research labs | 项目页正文 | ["https://arxiv.org/pdf/2310.08864v3"] |
| 参与学术实验室数 | 33 个 academic labs | DeepMind 博客 | ["https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |
| 参与机构数（摘要口径） | 21 institutions | arXiv 摘要 | ["https://arxiv.org/abs/2310.08864"] |
| 技能数（skills） | 527（博客约称 "more than 500"） | 摘要 / 博客 | ["https://arxiv.org/abs/2310.08864","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |
| 任务数（tasks） | 160266（博客约称 "150,000"） | 摘要 / 博客 | ["https://arxiv.org/abs/2310.08864","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |
| 统一数据格式 | RLDS (Reinforcement Learning Datasets) | SCARA 移植文 | ["https://arxiv.org/pdf/2409.03299"] |
| 统一动作接口 | 夹爪系 7-DoF（x,y,z,roll,pitch,yaw,gripper 或速率） | 项目页 | ["https://robotics-transformer-x.github.io/"] |
| RT-1 基座规模 | 35M 参数，15 帧历史，EfficientNet+USE | 论文 v7 | ["https://arxiv.org/pdf/2310.08864v7"] |
| RT-1-X 提升 | 比 Original Method/RT-1 高约 50%（小数据域） | 项目页 / 博客 | ["https://robotics-transformer-x.github.io/","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |
| RT-2-X 规模 | 约 55B 参数 | 项目页 | ["https://robotics-transformer-x.github.io/"] |
| RT-2-X 提升 | emergent skills 上约为 RT-2 的 3 倍 | 项目页 / 博客 | ["https://robotics-transformer-x.github.io/","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"] |

**后续影响（是否成为社区标准数据集 / 谁用了它）：**

- OpenVLA：970k 真实演示训练，代码内置 OXE 大规模训练支持，并以 OXE 为主要数据源之一。["https://arxiv.org/abs/2406.09246"]
- Octo：直接从 OXE 选 25 个数据集、800k 轨迹预训练。["https://octo-models.github.io/"]
- π0：方法学上明确采用 cross-embodiment training（异构机器人数据汇入同一模型）。["https://isaacbs.com/papers/pi0.pdf"]
- 2025 年仍在被扩展：OXE-AugE 对其中 16 个常用数据集做高质量增强，扩到 4.4M+ 轨迹，并在 OpenVLA、π0 上微调提升成功率。["https://arxiv.org/html/2512.13100"]
- 被多篇综述列为"统一数十个机器人数据集、用于大规模共训练"的标准数据集（1M+ trajectories / 527 skills）。["https://arxiv.org/html/2402.07127v3"]
- 结论：**OXE 已成为开源 VLA 事实上的标准预训练/评测数据底座**，是连接 Google 闭源 RT 系列与开源社区（OpenVLA、Octo、DROID 生态）的关键桥梁。

---

## 五、遗留问题汇总

1. **跨 embodiment 泛化的真实上限**：OXE/RT-X 证明了"共训练有正迁移"，但增益主要集中在同构桌面操作与小数据机器人；对形态/传感差异极大的平台（四足、移动操作、精细接触）泛化仍未解决。["https://arxiv.org/abs/2310.08864","https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"]
2. **离散动作 token 的分辨率瓶颈（RT-2-X）**：动作作文本 token 的离散化接口限制了精细连续控制分辨率，促使后续转向连续动作头（Octo 的 diffusion head → π0 的 flow-matching 动作专家）。["https://robotics-transformer.github.io/","https://arxiv.org/html/2405.12213v2"]
3. **缺乏互联网 VLM 预训练**：RT-1-X/RT-2-X 与 Octo 都不以大规模互联网视觉-语言预训练为核心（RT-2 虽共训网络数据，但 Octo 仅冻结 t5-base），开放词汇语义推理弱——这正是 OpenVLA（Llama2+DINOv2+SigLIP）要补的一环。["https://arxiv.org/html/2405.12213v2","https://arxiv.org/abs/2406.09246"]
4. **Octo 的能力边界**：腕相机融合差、规模仅 93M、评测限于桌面操作；适合作为"可快速微调的开源初始化"，但不具备大 VLA 的开放世界语义能力。["https://arxiv.org/html/2405.12213v2"]
5. **数据口径与数据质量**：OXE 由 60 个异构数据集拼接，标注语言稀疏、传感/频率不一（Octo 因此放弃微调语言编码器），数据 mixture 配比本身仍是未解的 scaling 问题。["https://arxiv.org/html/2405.12213v2","https://arxiv.org/pdf/2310.08864v3"]
6. **闭源权重 vs 开源权重的接力**：RT-2-X(55B) 权重并未完全开放给社区复现，Octo/OpenVLA 用小得多的开源模型逼近甚至超过它，说明"开源数据 + 开源架构"足以复刻闭源 VLA 能力——这是本环节最核心的遗产。["https://arxiv.org/abs/2406.09246","https://octo-models.github.io/"]
