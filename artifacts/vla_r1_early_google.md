# VLA 史前段调研笔记：Google 早期四作（SayCan / RT-1 / PaLM-E / RT-2）

> 主线位置：VLA 发展史主线（RT 系列 → 开源 VLA → π 系列）的第一环。
> 调研日期：2026-09-29。事实纪律：每条结论后附信源 URL；【未查证】标注无法从论文/官方材料直接核实之处。

---

## 0. 来源清单（URL 原样列出）

**arXiv 论文摘要页**
- RT-1 摘要页：https://arxiv.org/abs/2212.06817
- SayCan 摘要页：https://arxiv.org/abs/2204.01691
- PaLM-E 摘要页：https://arxiv.org/abs/2303.03378
- RT-2 摘要页：https://arxiv.org/abs/2307.15818

**arXiv 全文 HTML（ar5iv / arXiv 实验版 HTML，数字类结论主要取自此处）**
- RT-1 全文：https://ar5iv.labs.arxiv.org/html/2212.06817
- SayCan 全文（v2）：https://arxiv.org/html/2204.01691v2
- PaLM-E 全文：https://ar5iv.labs.arxiv.org/html/2303.03378
- RT-2 全文：https://ar5iv.labs.arxiv.org/html/2307.15818

**项目页 / 官方博客**
- RT-1 项目页：https://robotics-transformer1.github.io/
- SayCan 项目页：https://say-can.github.io/
- PaLM-E 项目页：https://palm-e.github.io/
- RT-2 项目页：https://robotics-transformer.github.io/
- Google DeepMind 官方博客（RT-2）：https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/

---

## 1. SayCan（2022-04，Google / Everyday Robots）

论文：*Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*（arXiv:2204.01691，v1 提交于 2022-04-04；v2 于 2022-08-16 加入 PaLM 结果）["https://arxiv.org/abs/2204.01691"]

### 1.1 它解决了什么问题
LLM 内部编码了大量关于世界的语义知识（" spill 了该怎么收拾"），但 LLM **没有任何具身经验**：它能讲出合理的叙事，却不知道当前这个机器人在这个厨房里**此刻能做什么**，直接让 LLM 输出动作会产出不可执行、与场景不符的计划 ["https://arxiv.org/abs/2204.01691"]。论文动机即"语言模型缺乏 real-world experience，难以在特定具身中做决策" ["https://arxiv.org/abs/2204.01691"]。

### 1.2 关键技术
- **架构（分层、非端到端）**：LLM（Say）负责高层任务分解/语义知识，机器人学来的技能 + 对应 value function（Can）负责世界落地。每一步，LLM 给候选技能打"语义可行"分 p_LLM，技能的语言条件 value function 给"当前场景可执行"分 p_affordance，两者乘积 p_combined = p_LLM × p_affordance 后选 argmax，自回归展开直到 "done" ["https://arxiv.org/html/2204.01691v2"]。
- **动作表征**：**不直接输出连续/离散关节动作**。输出是从一组预训练技能（技能 = 一个低层 policy + 一个 language-conditioned value function + 一句自然语言描述）里选一项；真正的低层动作由各技能自己的策略执行 ["https://arxiv.org/html/2204.01691v2"]。
- **LLM 主干**：540B PaLM（论文中称 PaLM-SayCan）；并消融了 PaLM 8B / 62B / 540B 与 FLAN-137B，整体随 LLM 变大而变好（540B 执行成功率 74%，8B 仅 38%）["https://arxiv.org/html/2204.01691v2"]。
- **数据策略**：为移动操作机器人提出 551 个候选技能（跨 7 个技能族、17 个物体），实际部署其中可组合、性能足够的子集 ["https://arxiv.org/html/2204.01691v2"]。低层技能/value function 的训练数据为 **68,000 条遥操作示范，10 台机器人，采集 11 个月**，另有约 12,000 条成功 episode 补充 ["https://arxiv.org/html/2204.01691v2"]。
- **效果数字**：101 条指令上，mock kitchen 规划成功率 84% / 执行成功率 74%；真实厨房 81% / 60%。**去掉 affordance 仅靠 LLM 时执行成功率为 0%**；行为克隆基线 BC-NL 仅 9%——这是"grounding 必要性"的直接证据 ["https://arxiv.org/html/2204.01691v2"]。错误归因：65% 来自 LLM、35% 来自 affordance ["https://arxiv.org/html/2204.01691v2"]。

### 1.3 遗产（对后续 VLA 的可复用范式）
- 确立了 **"LLM 提供语义先验 + 机器人模块提供接地"** 的分工范式，是后来一切 VLA 工作的思想原点。
- 证明了 **用乘积式打分把语言先验与可执行性融合**这一 grounding 思路有效；"把 LLM 输出约束在机器人已有技能集合内"成为后续分层具身系统的标准做法。
- 与 RT-1 同属 Everyday Robots 厨房场景车队，其运行日志（2912 条序列）后来直接成为 PaLM-E 移动操作规划模块的训练数据 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。

### 1.4 遗留的未解决问题
- **模块化、非端到端**：LLM 与低层 policy 是两个分离系统，误差在两层间累积；论文自己也指出 65% 错误来自 LLM 规划层 ["https://arxiv.org/html/2204.01691v2"]。
- **技能集合封闭**：机器人只能在预定义的几十个技能里做选择，学不会技能表之外的新动作；新技能需要单独训 value function。
- PaLM-E 论文进一步指出：affordance function 只能约束"此刻什么可做"，对需要长程组合、有大量不可行分支的 TAMP 任务信息不足——SayCan（即便给 oracle affordance 基线）在 TAMP 上得分为 0.0 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。

---

## 2. RT-1（2022-12，Robotics at Google / Everyday Robots）

论文：*RT-1: Robotics Transformer for Real-World Control at Scale*（arXiv:2212.06817，提交于 2022-12-13）["https://arxiv.org/abs/2212.06817"]

### 2.1 它解决了什么问题
计算机视觉/NLP/语音领域已能靠"大规模、任务无关的数据 + 大容量模型"实现 zero-shot 或小样本迁移，但**机器人领域一直做不到**，原因是真机数据采集极贵、泛化尤其关键 ["https://arxiv.org/abs/2212.06817"]。RT-1 要回答：能否像 NLP 那样，用一个**单一端到端模型**在大规模真机演示数据上训练，同时吞下海量多任务数据并实时闭环控制？

### 2.2 关键技术
- **架构（小而快，端到端）**：FiLM 条件化的 EfficientNet-B3（指令经 Universal Sentence Encoder 嵌入后，用 identity-initialized FiLM 层调制 ImageNet 预训练的卷积特征）→ TokenLearner（把 81 个视觉 token 软压缩成 8 个）→ 8 层 decoder-only Transformer，输出动作 token。全模型仅 **35M 参数**（其中图像 tokenizer 16M、Transformer 19M），以 **3 Hz 闭环**运行（推理 <100ms）["https://ar5iv.labs.arxiv.org/html/2212.06817"]。
- **动作表征**：这是首个把动作**离散化为 token**的机器人 transformer。动作 = 7 维臂部（x/y/z、roll/pitch/yaw、夹爪开合）+ 3 维底盘（x/y/yaw）+ 1 维模式切换（控臂/控底盘/终止），**每一维均匀分桶成 256 bins**，用标准类别交叉熵 + causal masking 训练 ["https://ar5iv.labs.arxiv.org/html/2212.06817"]。
- **数据策略**：**约 130,000 条真机演示，13 台 Everyday Robots 移动操作臂，采集 17 个月，覆盖 744 条语言指令（>700 任务）**，场景为办公室厨房"robot classrooms" ["https://ar5iv.labs.arxiv.org/html/2212.06817"]。另有把 QT-Opt 的 209k 条 Kuka 抓 bin 数据混入训练的跨本体吸收实验 ["https://ar5iv.labs.arxiv.org/html/2212.06817"]。**不使用任何网络图文预训练数据**——视觉主干只做过 ImageNet 分类预训练，语言只过 USE 编码器。
- **效果数字**：700+ 训练指令上 **97% 成功率**；3000 次真机试验评估泛化与鲁棒性；论文结果表中 RT-1 行在未见任务/扰动场景下约 59–83%，显著高于 BC-Z 等基线，但对全新物体/全新厨房的泛化明显掉点 ["https://ar5iv.labs.arxiv.org/html/2212.06817"]。

### 2.3 遗产
- **"动作离散成 token + Transformer 自回归建模"成为后续 VLA 的标准动作接口原型**：RT-2 直接沿用其 256-bin 分桶方案，只是把 token 塞进了大 VLM 的词表 ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- 证明了**机器人策略也能随数据规模/多样性 scaling**（论文系统消融了数据量、模型量、多样性对泛化的影响），为"机器人基础模型"路线立论 ["https://arxiv.org/abs/2212.06817"]。
- 高效 tokenization（FiLM-EfficientNet + TokenLearner）让 35M 小模型跑实时控制，与 PaLM-E/RT-2 的"大云模型"路线形成对照。

### 2.4 遗留的未解决问题
- **没有 web 语义先验**：主干只见过自家厨房的 13 万条轨迹，因此对训练分布外的**新奇物体、新奇符号（数字/图标）、抽象语义指令**泛化差——论文自己的未见任务/新厨房列就是证据 ["https://ar5iv.labs.arxiv.org/html/2212.06817"]。这正是 RT-2 要用 web 预训练 VLM 解决的痛点。
- **动作离散化（256 bins）的局限**：粗糙分桶带来动作精度损失、需逐维自回归采样、难以表达多模态动作分布——这一点是后续 π 系列改用 flow matching / 连续动作头的动机之一（本段只点出，不展开）。
- 仍依赖大规模**自采**数据（17 个月、13 台车），数据获取成本不可复制。

---

## 3. PaLM-E（2023-03，Google / Google DeepMind）

论文：*PaLM-E: An Embodied Multimodal Language Model*（arXiv:2303.03378，提交于 2023-03-06）["https://arxiv.org/abs/2303.03378"]

### 3.1 它解决了什么问题
SayCan 是"LLM + 外置技能模块"的两层系统；RT-1 是"有身体、无世界知识"的端到端小模型。PaLM-E 要回答：**能不能让 LLM 自己直接吃进连续传感器模态（图像、状态向量），在同一个模型里完成具身推理，而不是外挂一个 affordance 模块？** 论文明确指出现成的通用 VLM（如 PaLI）没见过具身数据，做不了具身推理；而 affordance grounding（SayCan 路线）在长程 TAMP 上信息不足 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。

### 3.2 关键技术
- **架构（embodied LLM）**：把图像/连续状态估计通过编码器（ViT，或 3D 感知的物体中心表征 OSRT）**投影进 LLM 的词嵌入空间**，与文本 token 一起进入自注意力；从预训练 LLM（PaLM）出发端到端训编码器。最大版本 **PaLM-E-562B = 540B PaLM + 22B ViT**，是当时报道的最大视觉-语言模型 ["https://arxiv.org/abs/2303.03378"] ["https://palm-e.github.io/"]。
- **动作表征**：**模型输出的是自然语言 token（高层子目标/子步骤），不是低层关节动作**。例如移动操作任务中 PaLM-E 逐字生成下一步计划文本（"1. Find a sponge..."），再由人映射到低层 policy（移动操作域直接复用 RT-1 的低层策略）；在 Language-Table 上以 1 Hz 输出语言子目标给 5 Hz 执行的低层 policy ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **三个机器人域**：TAMP（堆叠/抓取，96,000 训练场景）、Language-Table 桌面推挤、厨房移动操作（类 SayCan 设置）；移动操作规划用 SayCan 的 2912 条运行序列训练 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **数据策略 / 正迁移**：关键发现是**多任务联合训练带来正迁移**——把互联网规模的语言/图文/多模态任务与机器人任务混训，机器人任务数据效率大幅提升（TAMP 上只用 1% 数据、即每任务 320 条样本就显著优于单机器人训练），ViT-4B 模型靠 full mixture 规划性能翻倍以上；PaLI（无机器人数据）在机器人 VQA/规划上基本解不出，SayCan+oracle affordance 在 TAMP 上 0.0 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **通用能力保留**：PaLM-E-562B 在 OK-VQA 上达到 SOTA（无任务专属微调），并展现零样本多模态 chain-of-thought、少样本、跨图推理等涌现能力 ["https://arxiv.org/abs/2303.03378"]。

### 3.3 遗产
- **"把具身观察直接注入大 LLM 的词空间、让大模型本身做具身推理"**成为 RT-2 及后世 VLA 的架构母版：RT-2 直接以 PaLM-E 作为两个主干之一（RT-2-PaLM-E-12B）["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- 证明了**互联网图文预训练与机器人数据混训（co-training）对机器人有正迁移**——这正是 RT-2 "co-fine-tune 机器人数据 + web VQA 数据"训练配方的直接前驱 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- 物体中心/神经场景表征（OSRT）注入 LLM 的思路，影响了后续对空间表征与 tokenization 的讨论。

### 3.4 遗留的未解决问题
- **仍然只输出高层语言子目标，不直接输出低层动作**：闭环里还需要 RT-1/SayCan 式低层策略当"手"，不是一个端到端从图像到动作的 VLA。
- 562B 模型体积巨大、部署昂贵；论文也指出冻结 LLM 可行、但全量微调时 scale 增大才缓解灾难性遗忘 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- 它解决了"语义推理/规划的接地"，但**没有解决 RT-1 留下的"自采数据贵、动作精度受离散化限制"问题**——后者留给 RT-2（解决泛化）与 π 系列（解决动作表征）。

---

## 4. RT-2（2023-07，Google DeepMind）

论文：*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*（arXiv:2307.15818，提交于 2023-07-28）["https://arxiv.org/abs/2307.15818"]；官方博客同日发布 ["https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/"]

### 4.1 它解决了什么问题
RT-1 证明了端到端 transformer 策略可行，但模型只见过自家厨房 13 万条轨迹，**泛化受限于自采数据的分布**，无法获得 web 规模的语义知识（认识新物体、数字/符号、抽象指令、推理）["https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/"]。RT-2 要让**同一个端到端模型**既学会从机器人观测到动作的映射，又直接享受互联网规模视觉-语言预训练的红利 ["https://arxiv.org/abs/2307.15818"]。

### 4.2 关键技术
- **架构（VLA 一词的来源）**：直接拿已在 web 数据上预训练好的 SOTA VLM 当主干，co-fine-tune 后**不新增任何参数**就能输出动作。实例化两个：**RT-2-PaLI-X**（5B / 55B；ViT-22B 视觉 + 32B encoder-decoder）与 **RT-2-PaLM-E-12B**（ViT-4B 视觉投影 + PaLM decoder-only）["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- **动作表征（关键创新：把动作当"另一种语言"）**：沿用 RT-1 的分桶——连续维（6-DoF 位姿 + 夹爪）均匀分 256 bins，外加一个终止 token，共 8 个整数；再把这串数字拼成字符串（如 "1 128 91 241 5 101 127"），**当作普通文本 token 塞进 VLM 训练**：PaLI-X 直接复用其已有的整数 token，PaLM-E 则把词表中 256 个最低频 token 覆盖为动作 token（论文称之为 symbol tuning）。推理时再 detokenize 回关节命令 ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- **数据策略**：机器人轨迹沿用 RT-1 那套指令标注数据；web 侧是 PaLI-X/PaLM-E 的图文 mixture（主体 WebLI：约 10B 图文对筛到约 1B 高质量样本 + 各 caption/VQA 集）。**核心配方是 co-fine-tune（机器人数据与 web VQA 数据同批混训）而非只在机器人数据上微调**；机器人数据按采样权重占到训练 mixture 的约 50%（PaLI-X）/ 66%（PaLM-E）["https://ar5iv.labs.arxiv.org/html/2307.15818"]。训练目标即 next-token prediction（= 行为克隆损失）；解码时对机器人任务做输出约束——只在合法动作 token 上采样 ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- **效果数字**：**6,000 次真机评估**。RT-2 在泛化/推理类新指令上比 RT-1 等基线好约 2–3 倍，最佳 RT-2-PaLI-X 平均成功率超过次优基线 RT-1 三倍以上；涌现出符号理解（把东西放到指定数字/图标上）、推理（捡最小/最大物体、离某物体最近的）、人物识别（认出饮料罐上的名人）、多语言指令 ["https://arxiv.org/abs/2307.15818"] ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。消融显示：5B 模型 from scratch 平均泛化仅 9%，只做机器人微调 52%，**co-fine-tune 才到 63%**——直接证明 web 预训练是泛化来源 ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- **CoT 增强**：在数据里加一步自然语言 "Plan" 再跟 "Action"（如 "Plan: pick rxbar chocolate. Action: ..."），让 RT-2-PaLM-E 能做多步语义推理（即兴锤子=石头、困了=拿能量饮料）["https://arxiv.org/abs/2307.15818"] ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。

### 4.3 遗产
- **正式命名并定义了 "Vision-Language-Action (VLA)" 模型这一范畴**：VLM 主干 + 动作 token 化 + 机器人/web 数据 co-fine-tune，成为此后开源 VLA（OpenVLA、RT-2-X/OXE 系）与 π 系列共同遵循的范式骨架 ["https://arxiv.org/abs/2307.15818"]。
- **用 web 知识解决了 RT-1 的泛化痛点**：新奇物体、符号、抽象指令、推理能力全部来自互联网预训练，机器人数据只负责把感知对接到动作接口 ["https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/"]。
- "动作离散成字符串 token、复用 VLM 词表"成为最简动作接口；后续 Open X-Embodiment（RT-X）在此基础上扩到 22 种本体数据。

### 4.4 遗留的未解决问题
- **动作仍继承 RT-1 的 256-bin 离散 token**：精度有限、逐维自回归、单峰假设弱、难以表达多模态连续动作分布——这是后续 π 系列用 flow matching 直接回归连续动作的动机之一（此处只点出）。
- 模型大（12B–55B）、依赖云服务推理，实时性与边缘部署受限；机器人数据仍主要来自 RT-1 同一条自采车队，跨本体/跨场景多样性要等 OXE。
- 动作 token 是"塞进词表的符号"，与 VLM 原生语义空间的对齐仍较粗糙（论文自己靠覆盖最低频 token 解决）。

---

## 5. 演进关系：谁借鉴了谁、谁解决了谁的遗留问题

**时间线（以 arXiv 提交为准）**：SayCan（2022-04-04）→ RT-1（2022-12-13）→ PaLM-E（2023-03-06）→ RT-2（2023-07-28）["https://arxiv.org/abs/2204.01691"] ["https://arxiv.org/abs/2212.06817"] ["https://arxiv.org/abs/2303.03378"] ["https://arxiv.org/abs/2307.15818"]。
> 注意：任务书把 RT-1 列在最前，但按发表时间 **SayCan 其实早于 RT-1 约 8 个月**；二者是同一 Everyday Robots 车队并行推进的姐妹项目，并非 RT-1 之后才有 SayCan。

- **SayCan → PaLM-E**：PaLM-E 的移动操作域"largely follows SayCan 设置"，直接用 SayCan 的 2912 条运行序列训练规划模块，低层技能映射也沿用 Ahn et al. 2022 的定义 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。PaLM-E 批评 SayCan 的 affordance 只约束"此刻可做"、长程 TAMP 不够用（oracle affordance 基线在 TAMP 上 0.0），于是把"接地"从外置 value function 升级成"把传感器直接注入 LLM 词空间"，解决了 SayCan 的分层误差累积与封闭技能集问题 ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **RT-1 → PaLM-E**：PaLM-E 移动操作闭环里的**低层策略直接复用 RT-1**（高层 PaLM-E 输出文本子目标给 RT-1 式执行器）["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **RT-1 → RT-2（直接的"解决遗留问题"关系）**：RT-2 "building on the protocol proposed for RT-1, using a similar dataset, but expanding the model to use a large vision-language backbone"，动作分桶（256 bins）直接继承 RT-1 ["https://ar5iv.labs.arxiv.org/html/2307.15818"]。RT-1 的遗留痛点是**无 web 先验、泛化差**；RT-2 用 PaLM-E/PaLI-X 的 web 预训练 + co-fine-tune 把新奇物体/符号/推理泛化补了回来（消融：from-scratch 9% → co-fine-tune 63%）["https://ar5iv.labs.arxiv.org/html/2307.15818"]。
- **PaLM-E → RT-2**：RT-2 两个主干之一就是 PaLM-E 本体（RT-2-PaLM-E-12B）；PaLM-E "具身模态注入 LLM + web/机器人混训正迁移"的架构与数据思想，被 RT-2 收窄成"直接输出动作 token 的端到端策略"，把 PaLM-E 停留在"输出高层语言子目标"的那一步，进一步压到了"直接输出低层动作 token" ["https://ar5iv.labs.arxiv.org/html/2307.15818"] ["https://ar5iv.labs.arxiv.org/html/2303.03378"]。
- **共同未决 → 下游**：RT-1/RT-2 共用的 256-bin 离散动作 token 路线，其精度与多模态局限，构成后续 π 系列改用 flow matching 连续动作头的动机背景（本段不展开）。

---

## 6. 事实与推测区分说明

- **【已查证事实】**（均来自 arXiv 摘要页或 arXiv/ar5iv 全文正文，URL 见上）：四篇论文的提交日期、作者机构归属（Google Robotics / Everyday Robots / Google DeepMind）；RT-1 的 35M 参数、130k 演示、13 机器人、17 个月、744 任务、97% seen 成功率、256-bin 动作分桶、FiLM-EfficientNet-B3 + TokenLearner + 8 层 Transformer 架构、3Hz；SayCan 的 540B PaLM、551 候选技能、68k 演示/10 机/11 个月、101 指令上 84%/74%（mock）与 81%/60%（真实厨房）、无 affordance 0%；PaLM-E 的 562B=540B+22B、三机器人域、OSRT、正迁移/1% 数据（320 样本）、OK-VQA SOTA、输出语言子目标；RT-2 的 PaLI-X 5B/55B 与 PaLM-E-12B 主干、动作字符串 token 化（PaLI-X 复用整数 token / PaLM-E 覆盖 256 最低频 token）、WebLI 约 1B 样本、机器人数据占 50%/66%、6k 评估、2–3 倍泛化提升、from-scratch 9%/FT 52%/co-FT 63% 消融。
- **【推测/转述】**：① "RT-1 与 SayCan 是同车队姐妹项目"——二者作者名单与 Everyday Robots 平台高度重叠，论文互引，可合理判断，但"并行立项"这一组织细节【未查证】。② SayCan 论文中 551 是"提出的候选技能总数"，实际部署子集数量论文未在抓取片段中给出确切数字【未查证具体部署数】。③ RT-1 结果表中 73/76/83/59 各列的确切列名（未见物体/未见背景/扰动等）按论文泛化章节语境转述，精确列对应关系【未逐一核对表头】。④ "π 系列用 flow matching 解决离散动作局限"按任务书指定的演进叙事点出，未在本轮独立查证 π0 论文。
- **数字口径**：所有参数量/数据量/成功率均以 arXiv 论文正文为准；官方博客与论文一致的部分（如 RT-2 用 web+robot 数据训练）交叉引用了 DeepMind 博客，未发现与论文冲突的数字。
