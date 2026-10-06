# VLA 发展史主线调研（终点段）：开源 VLA → π 系列

> 调研范围：OpenVLA（2024）→ π0（2024）→ π0.5（2025）→ π0.7（2026，用户点名终点）。
> 时间基准：2026-09。事实纪律：每条结论后附信源 URL；【已查证事实】来自论文/官方博客，【媒体】为权威二手报道，查不到的数字一律标【未公开/未查证】，不做推测。
> 术语保留英文：VLA、VLM、flow matching、action chunk、embodiment、fine-tune、LoRA、RTC 等。

---

## 0. 来源清单

### 官方信源（第一信源）
- OpenVLA 论文（arXiv 2406.09246，2024-06-13 提交，v3 2024-09-05）：https://arxiv.org/abs/2406.09246
- OpenVLA 项目主页：https://openvla.github.io/
- π0 论文（arXiv 2410.24164，2024-10-31 提交，RSS 2025）：https://arxiv.org/abs/2410.24164
- π0 论文 HTML（v2，含架构细节）：https://arxiv.org/html/2410.24164v2/
- π0 开源公告（openpi，2025-02-04）：https://www.pi.website/blog/openpi
- π0.5 官方博客（2025-04-22）：https://www.pi.website/blog/pi05
- π0.7 官方博客（2026-04-16）：https://www.pi.website/blog/pi07
- π0.7 论文（arXiv 2604.15483v1，2026-04-16）：https://arxiv.org/html/2604.15483v1

### 二手/媒体与综述（仅补充，标注类型）
- RT-2 论文（动作 token 化对照基线）：https://arxiv.org/pdf/2307.15818
- RT-2 动作 token 说明（AI Wiki，转述论文）：https://aiwiki.ai/wiki/rt_2
- OpenVLA 动作 token 机制（NORA 论文转述）：https://arxiv.org/pdf/2504.19854v1.pdf
- OpenVLA 覆盖 256 least-used tokens（FailSafe 论文转述）：https://arxiv.org/html/2510.01642v1
- π0 数据规模（DeepLearning.AI 课程笔记，媒体）：https://charonhub.deeplearning.ai/p0-a-machine-learning-system-for-household-robotics/
- π0 数据规模与推理步数（AI Wiki，媒体/转述）：https://aiwiki.ai/wiki/pi_zero ；https://aiwiki.ai/wiki/pi0/raw
- π0 架构笔记（cloderic.com，媒体）：https://www.cloderic.com/content/2025-02-27-notes-on-pi0
- π0.7 媒体报道（AI Expert Magazine，媒体）：https://www.aiexpertmagazine.com/physical-intelligence-pi-0-7-model-composes-skills-fragmented-data/
- π0.7 媒体报道（Humanoids Daily，媒体）：https://www.humanoidsdaily.com/news/physical-intelligence-unveils-0-7-the-rise-of-compositional-generalization-in-robotics

> 说明：PI 官方博客实际域名为 `pi.website/blog`（非 physicalintelligence.company）；π0.7 官方 PDF 链接在博客页给出（https://www.pi.website/download/pi07.pdf ），与 arXiv 2604.15483 为同一内容。

---

## 1. OpenVLA（2024-06，Stanford / UC Berkeley 等）

### 1.1 解决了什么问题（承前）
- 在 OpenVLA 之前，已有的 VLA（以 Google RT-2 / RT-2-X 为代表）基本是**闭源、不可微调**的：RT-2-X 达 55B 参数，学术界与小团队拿不到权重，也没人系统研究如何高效把 VLA 微调到新机器人/新任务 ["https://arxiv.org/abs/2406.09246"]。
- OpenVLA 的定位是：把 VLA 做成一个**开源、可在消费级 GPU 上微调**的 7B 基座，让"预训练通用策略 → 下游微调"这条路可被社区复现和改进 ["https://arxiv.org/abs/2406.09246"]。
- 它直接继承 RT-2 的**离散动作 token** 路线（把动作当语言 next-token 预测），没有去解决离散动作表征本身的精度/多模态局限——这一局限要等到 π0 才被正面处理。

### 1.2 关键技术
- **架构**：7B 参数；视觉-语言主干基于 **Prismatic-7B**——Llama 2 语言模型 + 一个融合了 **DINOv2 与 SigLIP** 预训练特征的视觉编码器，在多粒度上提取视觉特征 ["https://arxiv.org/abs/2406.09246","https://arxiv.org/html/2504.19854"]。
- **动作表征（离散 token 路线）**：把连续机器人动作逐维**分箱（quantile-based）后映射到 LLaMA tokenizer 里最不常用的 256 个 token**，动作预测被 framing 成一个"vision-language" next-token 自回归任务 ["https://arxiv.org/html/2510.01642v1","https://arxiv.org/pdf/2504.19854v1.pdf"]。一次动作 = 一串离散 token，按语言模型方式自回归吐出。
- **数据策略**：在 **Open-X Embodiment** 上的 **97 万条（970k）真实机器人操作轨迹**上微调一个视觉条件语言模型；网络预训练来自互联网级 VLM 知识，机器人侧仅用开源 OXE 数据，没有自采私有数据引擎 ["https://arxiv.org/abs/2406.09246","https://openvla.github.io/"]。
- **多 embodiment / 灵巧操作**：开箱支持多种机器人；论文报告在 29 个任务、多机器人 embodiment 上以 7× 更少参数（7B vs 55B）**绝对成功率反超 RT-2-X 16.5%**，并比从零训练的 Diffusion Policy 高 20.4% ["https://arxiv.org/abs/2406.09246"]。
- **效率**：可用 LoRA 在消费级 GPU 上微调，可用量化（quantization）高效部署而不掉下游成功率；开放权重、微调 notebook 与 PyTorch 训练代码 ["https://arxiv.org/abs/2406.09246"]。

### 1.3 范式遗产
- 确立了"**开源 VLA = 开源 VLM 主干 + OXE 数据 + 动作 token 词表扩展**"的可复现范式，成为后续一大批开源工作（OpenVLA-OFT、NORA、SmolVLA 等）的事实基座 ["https://arxiv.org/pdf/2512.22539v2","https://arxiv.org/html/2504.19854"]。
- 证明了 7B 量级 + LoRA 微调路线的可行性，把 VLA 从大厂 demo 拉进了实验室可复现区间。

### 1.4 遗留的未解决问题
- **离散 token 的固有局限**：逐维分箱 + 自回归生成，在高频、高精度、多模态的灵巧操作上精度受限；动作被"平均"到某个 bin，且自回归逐 token 解码在高频控制下延迟高——这正是后来 OpenVLA-OFT 改回归头、π0 改 flow matching 的动机 ["https://arxiv.org/html/2511.18960v2","https://arxiv.org/pdf/2512.22539v2"]。
- 评测仍高度分布于训练相近的场景；跨环境/跨 embodiment 的开放世界泛化弱（π0.5 博客明确把"π0 及近期 VLA 都在与训练相近的环境评测"列为待解问题）["https://www.pi.website/blog/pi05"]。

---

## 2. π0 / pi-zero（2024-10-31，Physical Intelligence）

> 论文：*π0: A Vision-Language-Action Flow Model for General Robot Control*，arXiv 2410.24164，RSS 2025 ["https://arxiv.org/abs/2410.24164"]。

### 2.1 解决了什么问题（承前）
- 承 RT-2 / OpenVLA 的**离散动作 token** 路线：离散分箱在**高频、灵巧、双手操作**（叠衣服、擦桌子、装箱）上精度不足，且自回归逐 token 解码难以刻画动作的**多模态分布**（同一物体常有多种合理抓法）["https://arxiv.org/pdf/2410.24164.pdf?_sp=fe5e3699-dc47-4a51-b678-d7cef4b61754","https://aiwiki.ai/wiki/pi_zero"]。
- π0 的目标：用一个网络同时吃互联网级 VLM 语义知识，又能输出**连续、高频、多模态**的机器人动作，并跨单臂/双臂/移动操作机器人泛化 ["https://arxiv.org/abs/2410.24164"]。

### 2.2 关键技术
- **架构（双专家、单 transformer，受 Transfusion 启发）**：
  - VLM 主干用开源 **PaliGemma（3B 参数）**，继承互联网规模语义知识 ["https://arxiv.org/html/2410.24164v2/"]。
  - 新增一个从零初始化的 **action expert（动作专家，约 300M 参数）**，总计约 **3.3B 参数** ["https://arxiv.org/html/2410.24164v2/","https://aiwiki.ai/wiki/physical_intelligence/raw"]。
  - 单一 transformer 用多目标训练：离散 token 走 cross-entropy（语言），连续动作 token 走 flow matching loss（受 Transfusion 多目标训练启发）["https://arxiv.org/pdf/2410.24164.pdf?_sp=fe5e3699-dc47-4a51-b678-d7cef4b61754"]。
- **动作表征（flow matching 连续动作块，本主线的关键转折）**：
  - 不再把动作分箱成离散 token，而是用 **conditional flow matching** 学一个向量场，把高斯噪声"积分"成一段未来动作轨迹；一次预测一个 **H=50 步的 action chunk**（约 1 秒 @ 50Hz），而非单步动作 ["https://aiwiki.ai/wiki/pi_zero","http://www.thecprogramminglanguage.com/pi0.html"]。
  - 推理时用约 **10 步积分（去噪）** 把噪声 action chunk 解码成真实轨迹（【媒体/转述】，见 openpi 复现文档与 AI Wiki）["https://aiwiki.ai/wiki/pi0/raw","https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/4/"]。
  - 为统一不同关节数/夹爪/指令模式的机器人，把 state/action 向量 padding 到训练集中最大机器人维度（公开配置约 18 维）【媒体/转述】["https://aiwiki.ai/wiki/pi_zero"]。
  - flow matching 相比离散 token：直接建模连续动作分布，能刻画多模态（多种合法策略），轨迹更平滑，适配高频灵巧任务；代价是推理要跑多步去噪/积分，比纯自回归离散解码更重。
- **数据策略**：
  - 私有自采数据约 **9.03 亿 timesteps（约 1 万小时）**，覆盖 **68 个任务、7 种机器人配置**（从单臂到双臂+移动底座）【媒体/转述】["https://www.cloderic.com/content/2025-02-27-notes-on-pi0","https://charonhub.deeplearning.ai/p0-a-machine-learning-system-for-household-robotics/","https://aiwiki.ai/wiki/pi_zero"]。
  - 叠加公开数据：**Open-X Embodiment、Bridge v2、DROID**【媒体/转述】["https://aiwiki.ai/wiki/pi0/raw"]。
  - 路线是"**互联网 VLM 预训练知识 + 大规模多机器人自采机器人数据微调**"。
- **多 embodiment 与双手/灵巧**：单臂、双臂、移动操作统一在一个模型里；展示叠衣服、擦桌子、组装箱子等灵巧/双手任务，预训练后 zero-shot 即可执行，并能通过微调获得新技能 ["https://arxiv.org/abs/2410.24164"]。
- **开放权重**：2025-02-04 以 openpi 仓库（Apache 2.0，含 Gemma 许可）开源 π0 代码与权重，并同时发布自回归版 **π0-FAST** ["https://www.pi.website/blog/openpi"]。

### 2.3 范式遗产
- 开创了"**预训练 VLM 主干 + flow matching action expert**"这一后续 π 系列（π0.5/π0.6/π0.7）一以贯之的架构母版；把"连续动作块 + 少步去噪"确立为高频灵巧 VLA 的主流动作解码方式之一（与离散 AR、扩散并列）["https://arxiv.org/pdf/2512.22539v2"]。
- 证明一个 3.3B 的模型就能在多种机器人上做灵巧操作，把"机器人基座模型"从 55B 闭源路线拉到小而强、可微调的方向。

### 2.4 遗留的未解决问题
- 评测仍在与训练相近的环境中；**开放世界泛化**（全新厨房/卧室、杂乱家庭）不足——这是 π0.5 直接要补的洞 ["https://www.pi.website/blog/pi05"]。
- flow matching 多步去噪带来推理延迟（第三方实测 π0 类 flow 策略在 RTX 4090 上约 300–600ms、控制频率仅 1.5–3Hz，【媒体】）["https://vnrobo.com/en/blog/lerobot-g1-pi0fast-whole-body"]；这也是后来 π0.7 做 RTC、推理优化的动因。
- 长时程任务、显式高层规划、跨任务"组合泛化"尚未出现。

---

## 3. π0.5（2025-04-22，Physical Intelligence）

> 官方博客：*π0.5: a VLA with Open-World Generalization*（2025-04-22）["https://www.pi.website/blog/pi05"]。

### 3.1 解决了什么问题（承前）
- π0 及当时其他 VLA 都在**与训练分布接近的环境**里评测；π0.5 要解决的是**开放世界泛化**：把移动操作机器人放进**训练数据里完全没见过的家庭**，完成收拾厨房/卧室、铺床等长时程任务 ["https://www.pi.website/blog/pi05"]。
- 官方明确：π0.5 的目标**不是**新增高灵巧技能，而是"泛化到新场景"；它"远非完美"，常在高层语义推断和底层动作上犯错 ["https://www.pi.website/blog/pi05"]。

### 3.2 关键技术
- **架构**：基于 π0 VLA 扩展；同一个模型里**既有离散自回归解码、又有 flow matching 连续解码**——离散路径用于输出高层子任务（语言，如"pick up the pillow"），flow matching 路径用于输出底层电机指令（一个 **50 步 / 1 秒**的连续 action chunk）；action expert 约 **300M**（与 π0 同量级）["https://www.pi.website/blog/pi05"]。
- **高低层"思维链"式推理**：先让模型产出文本化的高层 action，再据此用 flow matching 专家选底层连续动作；同一模型同时承担高层决策与底层运动控制，思路延续 PI 的 Hi Robot 系统 ["https://www.pi.website/blog/pi05"]。
- **核心数据策略 = 异构数据 co-training**：在多源数据上混合训练——
  - **WD（Web Data）**：图像 captioning、VQA、目标检测等互联网多模态数据；
  - **ME（Multiple Environment）**：用非移动机器人进入大量不同家庭采的数据；
  - **CE（Cross Embodiment）**：π0 原训练集里的跨 embodiment 数据；
  - 还有"口头指导"演示（人逐步用自然语言带机器人）与高层子任务标注 ["https://www.pi.website/blog/pi05"]。
- **消融（官方，OOD = 未见过的新家）**：
  - 完整 π0.5：OOD follow rate **94%** / OOD success **94%**；
  - 去掉 WD：80% / 74%；去掉 CE：67% / 49%；去掉 ME：33% / 31% ["https://www.pi.website/blog/pi05"]。
  - 结论：**Web Data 对 OOD 物体识别帮助最大，来自其他机器人的数据（ME+CE）在所有条件下都关键** ["https://www.pi.website/blog/pi05"]。
- **scaling**：随训练环境数增加，泛化单调上升；**约 100 个训练环境**后，性能逼近"直接在测试环境训练"的上界基线；"无 ME/CE"版本只剩约 **400 小时**同机器人移动操作数据 ["https://www.pi.website/blog/pi05"]。

### 3.3 范式遗产
- 确立了"**机器人动作数据 + 互联网多模态数据 + 高层子任务标注 co-training**"是开放世界泛化的关键配方；把"高层语言子任务 + 底层 flow matching 动作"的双层结构固化为 π 系列后续架构（π0.7 的高层语义政策即由此发展）["https://www.pi.website/blog/pi05"]。

### 3.4 遗留的未解决问题
- 灵巧/新技能仍需微调才能到最好水平；长时程任务常失败；高层语义推断与底层动作都会出错 ["https://www.pi.website/blog/pi05"]。
- 跨 embodiment 的大形态差距上表现差（π0.7 论文显示 π0.5 在大 embodiment gap 任务上显著退化）["https://arxiv.org/html/2604.15483v1"]。
- 数据里没有"失败/次优数据"的系统性利用，也没有显式视觉子目标/策略元数据——这是 π0.7 要补的。

> 备注：π0.5 之后、π0.7 之前还有 **π0.6（约 2025-11）**，走 RL 后训练（Recap 算法）刷单任务成功率/吞吐，产出 π*0.6 专家模型；π0.7 论文明确"builds on π0.6-MEM 架构"并与之对比 ["https://arxiv.org/html/2604.15483v1","https://hokai.io/hub/companies/physical-intelligence"]。π0.6 非本调研点名对象，仅作衔接。

---

## 4. π0.7 深度专项（2026-04-16，Physical Intelligence）

> 第一信源：官方博客 *π0.7: a Steerable Model with Emergent Capabilities*（2026-04-16）["https://www.pi.website/blog/pi07"] + 论文 *π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities*（arXiv 2604.15483v1，2026-04-16）["https://arxiv.org/html/2604.15483v1"]。

### 4.1 发布信息【官方】
- **名称确为 π0.7**，发布时间 **2026-04-16**，与博客同日挂 arXiv（2604.15483）["https://www.pi.website/blog/pi07","https://arxiv.org/html/2604.15483v1"]。
- 官方定位：一个 **steerable（可引导）的通用机器人基座模型**，出现"step-change in generalization"，首次展示机器人**组合泛化（compositional generalization）**的早期迹象——把不同任务学到的技能重新组合去做训练里没见过的事（如操作新厨房电器、在从未叠过衣服的新机器人上叠衣服）["https://www.pi.website/blog/pi07"]。

### 4.2 架构【官方】
- **总参数约 5B**，构成为：
  - **VLM 主干 4B**（初始化自 **Gemma3 4B**，含一个 **400M 视觉编码器**）；
  - **MEM 风格的 video/历史编码器**（对历史观测做时间+空间压缩，任意帧数压成固定 token 数）；
  - **flow matching action expert 860M** ["https://arxiv.org/html/2604.15483v1"]。
  - （媒体对"约 5B"的转述与官方一致）【媒体】["https://www.aiexpertmagazine.com/physical-intelligence-pi-0-7-model-composes-skills-fragmented-data/"]。
- **是否升级主干**：是——从 π0/π0.5 的 PaliGemma 3B 升级到 **Gemma3 4B**；动作专家从 300M 增大到 **860M**，并引入 MEM 历史编码器与视觉子目标条件 ["https://arxiv.org/html/2604.15483v1"]。
- **flow matching 是否延续**：是。action expert 仍是 flow matching：860M transformer，固定 **50 个 action token = 50 步 action chunk**，token 间双向注意力、并交叉注意 VLM 主干激活；用 adaptive RMSNorm 注入 flow matching 时间步信息 ["https://arxiv.org/html/2604.15483v1"]。
- **是否有显式规划器/分层**：有"分层"但不是独立大规划器——
  - 运行时由一个**基于同一架构的高层语义政策**输出语言子任务（subtask instruction）；
  - 子目标图像由一个**轻量 world model** 生成（初始化自 **BAGEL 14B** mixture-of-transformers：7B ViT LLM 主干 + 7B 生成主干，用 CFM 损失训练，每 **Δ=4 秒**重生成一次子目标）["https://arxiv.org/html/2604.15483v1"]。
  - 即：高层语言子任务 + world-model 子目标图像，共同"steer"底层 flow matching 策略，而非一个重型显式 planner。
- **知识绝缘（KI）训练配方**：VLM 主干用 FAST token 做监督；action expert 虽然注意主干全部激活，但**梯度不回传到主干**，主干只受相对稳定的离散 cross-entropy 损失训练——保护预训练知识不被机器人梯度破坏 ["https://arxiv.org/html/2604.15483v1"]。
- **观测/状态编码**：最多 4 路相机（前视 + 双腕 + 可选后视），每路最多 6 帧历史（步幅 1 秒），最多 3 张子目标图；图像 resize 到 448×448；block-causal 注意力（观测/子目标双向、文本 causal）；本体状态 q_t 改用**线性投影**嵌入（不同于 π0.6 用离散文本 token 表示）["https://arxiv.org/html/2604.15483v1"]。
- **RTC**：训练时仿真 0–12 步推理延迟（对应 50Hz 机器人上最大 240ms 延迟），即 training-time real-time action chunking，保证延迟下动作平滑 ["https://arxiv.org/html/2604.15483v1"]。

### 4.3 训练数据规模与构成【官方】
- **构成**（论文 VI-A，官方定性描述）：
  - 大量不同机器人平台的演示数据（静态/移动、单臂/双臂，实验室类与家庭类、野外家庭）；
  - **大量策略评测产生的自主（autonomous）数据**；
  - 人在策略 rollout 中的介入（interventions）；
  - **开源机器人数据集**（文中举例 DROID）；
  - **第一人称人类视频（egocentric human video）**；
  - 互联网侧非机器人辅助数据（目标定位、属性预测、VQA、纯文本、视频 captioning）["https://arxiv.org/html/2604.15483v1"]。
- **显著区别于经典 VLA 的做法**：重度使用**次优/失败数据**——包括低质量演示（失败段、带错的成功段）以及 **π*0.6 RL 后训练模型在评测中收集的数据**，相当于把 RL 专家的行为蒸馏进通用模型；靠 episode metadata 标注来"区分"这些不同质量的数据 ["https://arxiv.org/html/2604.15483v1"]。
- **数据总量（小时数/timestep 数）**：【未公开/未查证】——官方博客与论文只给构成与消融，未披露 π0.7 的总训练小时数/timestep 数。
- **核心创新 = diverse context / prompt conditioning**：给每段数据加**多种模态的提示**，不仅说"做什么"还说"怎么做"——子任务语言指令、子目标图像、episode 元数据（速度按 500 步分箱、质量 1–5 分、是否出错标注）、控制模式（joint/ee）；训练时对每个 prompt 组件随机 dropout（子目标图仅 25% batch、有子目标时 30% 丢掉子任务、元数据整体 15% 全丢 + 各组件 5%），使测试时可灵活组合任意子集 ["https://arxiv.org/html/2604.15483v1","https://www.pi.website/blog/pi07"]。

### 4.4 能力边界【官方】
- **开箱即用的专家级灵巧**：无需任务专项后训练，即可做意式咖啡机、叠衣服、取垃圾袋、折箱子、削皮等，并在**成功率与吞吐上追平甚至超过**用 Recap/RL 专项训练的 π*0.6 专家（靠把 RL 经验蒸馏进通用模型）["https://www.pi.website/blog/pi07","https://arxiv.org/html/2604.15483v1"]。
- **指令泛化**：在 **4 个未见厨房 + 2 个未见卧室**里执行 3–6 步指令序列；在复杂/非常规指称（如"pick up an object I would use to eat soup"）上显著优于 π0.5/π0.6；用 world-model 生成的子目标图进一步提升 ["https://arxiv.org/html/2604.15483v1"]。
- **跨 embodiment**：把用小型静态双臂机器人采的**叠 T 恤**任务，zero-shot 迁移到**双臂 UR5e**（该硬件上**没有任何叠衣服数据**），成功率达到"首次在 UR5e 上尝试该任务的专家人类操作员"水平（10 名操作员、约占操作团队前 2%、平均 375 小时遥操经验）；且策略会自发发现适配新形态的新抓法（如 UR5e 上改用垂直抓握而非源机器人的倾斜抓握）["https://www.pi.website/blog/pi07","https://arxiv.org/html/2604.15483v1"]。
- **组合泛化**：zero-shot 提示操作**空气炸锅烤红薯**只能部分完成；改用逐步**语言 coaching** 后明显变好；再用 coaching 数据**微调一个高层政策**，即可完全自主执行、无需任何遥操作；同类任务还有卸载空气炸锅、烤贝果 ["https://www.pi.website/blog/pi07","https://arxiv.org/html/2604.15483v1"]。
  - 媒体补充（【媒体】）：称 zero-shot 空气炸锅成功率约 5%，经约 30 分钟"人写提示词/语言引导"后升到约 95%——官方正文未给出这两个具体百分比数字，仅定性描述"部分完成 vs coaching 后显著变好"，故百分比以媒体转述对待 ["https://www.aiexpertmagazine.com/physical-intelligence-pi-0-7-model-composes-skills-fragmented-data/"]。
- **抗数据集偏置**：在"Reverse Bussing""Reverse Fridge→Microwave"等违背数据常见做法的任务上，π0.7 显著优于前代，说明其真的听语言指令而非盲目复制数据分布；其中"Reverse Fridge→Microwave"必须靠生成子目标图（π0.7 (GC)）才成功 ["https://arxiv.org/html/2604.15483v1"]。
- **多臂协作**：以双臂（bimanual）灵巧为主（叠衣、装箱、意式咖啡），也覆盖单臂/移动操作；**未见"多机器人协作（multi-robot coordination）"**的公开演示——【未公开/未查证】。

### 4.5 与 π0.5 的具体差异【官方】
- **主干升级**：PaliGemma 3B → Gemma3 4B；动作专家 300M → 860M；新增 MEM 历史编码器与子目标图像条件 ["https://arxiv.org/html/2604.15483v1"]。
- **从"新环境泛化"到"新任务组合泛化"**：π0.5 主攻在没见过的家里做**已知类**清洁任务；π0.7 进一步要求做**训练里没见过的新任务**（新电器、新物体玩法），并开箱达到专家级灵巧 ["https://www.pi.website/blog/pi07","https://www.pi.website/blog/pi05"]。
- **数据利用**：π0.5 的 co-training 仍以高质量演示 + web 数据为主；π0.7 系统性纳入**失败/次优自主数据 + RL 专家蒸馏 + 第一人称人类视频**，并用元数据消歧混合质量数据 ["https://arxiv.org/html/2604.15483v1"]。
- **引导方式**：π0.5 主要靠文本指令 + 高层子任务；π0.7 增加**子目标图像、速度/质量/出错元数据、控制模式**等多模态"steer"信号，测试时可由 world model 自动出子目标 ["https://arxiv.org/html/2604.15483v1"]。
- **大 embodiment gap 上**：π0.5 在大形态差距任务上显著退化，π0.6 仍可，π0.7 在最大差距（小型静态双臂 → 单臂 UR5e、小型双臂 → 大型双臂 UR5e 叠衣）上**显著优于前代** ["https://arxiv.org/html/2604.15483v1"]。

### 4.6 已知局限 / 失败模式【官方】
- **zero-shot 成功率仍明显低于分布内**：见得多的任务成功率常 >90%，而**未见任务/未见任务-机器人组合成功率落在 60–80% 区间**（论文 Discussion 原话）["https://arxiv.org/html/2604.15483v1"]。
- **长时程未见任务不能纯 zero-shot**：如"烤红薯"这类约 5 分钟、多阶段的任务，直接 prompt 不work，必须靠语言 coaching，或用 coaching 数据训出高层政策后才能自主 ["https://arxiv.org/html/2604.15483v1"]。
- **"见过/没见过"难以界定**：数据太大太杂，很难确证某个任务到底是不是真"unseen"——这是实验上的一个诚实局限 ["https://arxiv.org/html/2604.15483v1"]。
- **推理开销**（官方 Appendix A-D）：π0.7 + 高层政策跑在单张 H100 上；最小配置（3 相机、5 步去噪、训练时 RTC）**38ms**，但开启 MEM 编码器 + 子目标图后最坏 **127ms**；world model 生成子目标图要 4×H100 张量并行 + 8-bit 量化 + SageAttention，25 步去噪约 **1.25 秒**，靠异步流水线掩盖延迟 ["https://arxiv.org/html/2604.15483v1"]。
- **开放权重/具体 benchmark 分数卡**：截至本调研，π0.7 是否开放权重、是否有标准化 benchmark 逐项数字表——【未公开/未查证】（官方博客强调招聘/合作，未宣布开源）。

---

## 5. 动作表征演进线（专节）：离散 token → 连续 flow matching → π 系列后续

> 这条线是本主线的核心叙事：**每一步解决了什么、代价是什么**。

### 5.1 RT-2（2023）：把动作"当语言"——离散 token 自回归
- 做法：把每个连续动作维度**均匀分箱成 256 个 bin**，每 bin 映射到词表里一个 token（PaLI-X 直接复用 0–255 的整数 token）；一个动作 = terminate 标志 + 6-DoF 末端增量 + 夹爪状态，拼成一串 token，用 next-token 自回归吐出，再映射回 bin 中心值 ["https://arxiv.org/pdf/2307.15818","https://aiwiki.ai/wiki/rt_2"]。
- **解决了什么**：让机器人控制**复用一个已预训练好的 VLM**，把互联网语义知识直接迁移进控制；动作变"语言"，无需为动作单独造解码头 ["https://arxiv.org/pdf/2307.15818"]。
- **代价/局限**：
  - 连续动作被**量化到离散 bin**，精度受 bin 粒度限制；
  - 自回归逐 token 解码**慢、高频控制延迟大**；
  - 难以刻画动作的**多模态分布**（同一状态多种合理动作），常把多种策略"平均"成一种。

### 5.2 OpenVLA（2024-06）：离散 token 路线的开源化
- 做法：沿用 RT-2 思路，但把动作 token 写进 LLaMA tokenizer 里**最不常用的 256 个 token**，用 quantile 分箱，在 OXE 970k 轨迹上 next-token 预测 ["https://arxiv.org/html/2510.01642v1","https://arxiv.org/pdf/2504.19854v1.pdf"]。
- **解决了什么**：把这条离散路线**开源、7B 化、可 LoRA 微调**，让社区能复现、改进；证明小 7× 的模型可反超 55B 闭源 RT-2-X ["https://arxiv.org/abs/2406.09246"]。
- **代价**：**继承了离散 token 的全部固有局限**——高频灵巧操作精度不足、自回归解码慢、多模态动作建模弱；它是"离散路线的开源巅峰"，而非动作表征的革新。

### 5.3 π0（2024-10）：flow matching 连续动作块——范式转折
- 做法：不再分箱；用 **conditional flow matching** 学向量场，把高斯噪声积分成一段 **H=50 步连续 action chunk**，约 10 步去噪解码；VLM 主干负责语义，独立的 action expert 负责连续动作生成（Transfusion 式多目标单 transformer）["https://arxiv.org/pdf/2410.24164.pdf?_sp=fe5e3699-dc47-4a51-b678-d7cef4b61754","https://aiwiki.ai/wiki/pi_zero"]。
- **解决了什么**：
  - **连续动作** → 摆脱 bin 量化精度损失；
  - **显式建模多模态动作分布** → 适合灵巧/双手/高频任务；
  - **action chunk + 少步去噪** → 轨迹平滑、比逐维自回归更适合高频控制。
- **代价**：
  - 推理要跑**多步去噪/积分**，比单步离散解码**更重、延迟更高**（第三方实测 flow 策略延迟可达数百 ms）["https://vnrobo.com/en/blog/lerobot-g1-pi0fast-whole-body"]；
  - 引入额外 action expert 参数与训练复杂度；
  - 需要大规模多机器人自采数据（约 1 万小时）才能喂出泛化，数据门槛高。

### 5.4 π0.5 / π0.6 / π0.7：在 flow matching 母版上"加引导、加记忆、加数据多样性"
- **π0.5**：保留 flow matching 连续动作 expert（50 步/1 秒），**新增离散自回归路径输出高层语言子任务**——形成"高层语言推理（离散 AR）+ 底层连续动作（flow matching）"双层；动作表征本身仍是 flow matching，但用 co-training 异构数据解决开放世界泛化 ["https://www.pi.website/blog/pi05"]。
- **π0.6**（衔接）：在 flow matching 基座上做 **RL 后训练（Recap）**，优化鲁棒性与吞吐，产出 π*0.6 专家；把"强化学习"这条加进动作优化，但不改变 flow matching 动作表征 ["https://arxiv.org/html/2604.15483v1"]。
- **π0.7**：flow matching 延续并放大（860M expert、50 步 chunk、RTC），核心新增**多模态 context conditioning**——子任务语言、子目标图像、速度/质量/出错元数据、控制模式；并用**训练时 RTC** 与推理优化把最小延迟压到 38ms、最坏 127ms，部分对冲 flow matching 的延迟代价；同时把 π*0.6 RL 专家的经验**蒸馏**进通用模型，让一个模型顶多个专家 ["https://arxiv.org/html/2604.15483v1"]。
- **这一段的净效果**：flow matching 从"新动作表征"变成 π 系列默认底座；创新点上移到**如何喂更多样/更次优的数据、如何用 prompt/子目标/元数据引导策略、如何蒸馏 RL 专家、如何压低推理延迟**——动作表征本身稳定了，战场转向数据与引导。

### 5.5 一句话总结演进
> RT-2/OpenVLA 用**离散 token** 把控制变成语言生成（复用 VLM 知识，但精度/延迟/多模态受限）；π0 用 **flow matching 连续动作块** 换高频多模态灵巧（代价是多步去噪、推理更重、数据门槛更高）；π0.5→π0.7 在 flow matching 底座上叠加**高低层分层、world-model 子目标、元数据引导、失败/RL 数据蒸馏与 RTC 延迟优化**，把"动作怎么表示"的问题基本收敛，转向"如何用更多样数据 + 更强引导实现组合泛化"。

---

## 6. 遗留问题汇总（全主线）
1. **zero-shot 泛化成功率仍偏低**：π0.7 官方承认未见任务/未见任务-机器人组合成功率仅 60–80%，远低于分布内 >90% ["https://arxiv.org/html/2604.15483v1"]。
2. **长时程未见任务不能纯 prompt**：多阶段、数分钟级任务仍需语言 coaching 或先微调高层政策，尚不能 LLM 式"零样本组合" ["https://arxiv.org/html/2604.15483v1"]。
3. **flow matching 推理延迟**：虽经 RTC/量化优化，带 MEM + 子目标仍达 127ms，子目标 world model 1.25s、需 4×H100；高频全身/人形控制仍吃紧 ["https://arxiv.org/html/2604.15483v1","https://vnrobo.com/en/blog/lerobot-g1-pi0fast-whole-body"]。
4. **"见过/没见过"不可证**：数据规模过大，论文自承难以严格界定真正的泛化 vs 记忆 ["https://arxiv.org/html/2604.15483v1"]。
5. **数据规模不透明**：π0.7 总训练数据量【未公开/未查证】；π0 的约 1 万小时/903M timestep 为【媒体/转述】级别，PI 官方论文未在我核到的正文里给出逐字数字。
6. **多机器人协作 / 人形全身控制**：π 系列公开能力集中在双臂桌面/移动操作；多臂协作、人形全身（WBC）公开证据【未公开/未查证】。
7. **开源程度不一**：OpenVLA、π0（openpi）开源权重；π0.5/π0.7 截至本调研未宣布开放权重，社区复现门槛仍高 ["https://www.pi.website/blog/openpi","https://www.pi.website/blog/pi07"]。

---

*笔记完成时间：2026-09-29。所有 URL 均来自检索工具返回原文；数字类结论已区分官方与媒体转述；未查到项一律标注【未公开/未查证】。*
