# VLA 发展史补充：同期全景模型与评估基准演进（R4 调研笔记）

> 定位：本笔记为主线（RT 系列 → 开源 VLA → π 系列）提供同期生态补充与评测维度支撑，不展开主线技术细节。所有技术事实后附可追溯 URL；无法核实者显式标注【未查证】。

---

## 一、来源清单

| # | 信源 | 用途 |
|---|---|---|
| S1 | arXiv:2503.14734 — GR00T N1 论文 | NVIDIA 官方论文 |
| S2 | NVIDIA Research 官方页 — GR00T N1 发布页 | 发布时间/定位 |
| S3 | arXiv:2406.04339 — RoboMamba 论文（ar5iv 全文） | 架构/动作表征/参数 |
| S4 | crossformer-model.github.io — CrossFormer 项目页 | CrossFormer 真实标题/作者/数据规模 |
| S5 | Google DeepMind 博客 — Gemini Robotics On-Device（2025-06-24） | Gemini Robotics 家族时间线 |
| S6 | Google 官方博客 — Gemini Robotics 首发（2025-03-12） | 首发定位 |
| S7 | arXiv:2510.03342 — Gemini Robotics 1.5 技术报告 | GR 1.5 能力边界 |
| S8 | arXiv:2410.07864 — RDT-1B 论文 | 清华 RDT-1B 架构 |
| S9 | arXiv:2306.03310 — LIBERO 论文 | LIBERO 任务套件 |
| S10 | UT Austin RPL 主页 — LIBERO NeurIPS 2023 | 发表venue |
| S11 | arXiv:2405.05941 — SIMPLER 论文 | 仿真-真实一致性 |
| S12 | simpler-env.github.io — SIMPLER 项目页 | 作者列表/覆盖硬件 |
| S13 | arXiv:2310.08864 — Open X-Embodiment / RT-X | 跨具身数据/评测 |
| S14 | robotics-transformer-x.github.io — RT-X 项目页 | 关键数字 |
| S15 | Google DeepMind 博客 — RT-X 发布（2023-10-03） | RT-1-X +50% |
| S16 | arXiv:2406.09246 — OpenVLA 论文附录 | Google Robot 12 tasks / Bridge 17 tasks 评测细节（RT-Bench 体系可查证部分） |
| S17 | arXiv:2410.24164 — π0 论文 | π0 评测任务 |
| S18 | pi.website/blog/pi05 — Physical Intelligence 官方博客 | π0.5 评测方式 |
| S19 | pi.website/blog/pi07 — Physical Intelligence 官方博客 | π0.7 评测方式 |
| S20 | arXiv:2604.15483 — π0.7 技术报告 | π0.7 out-of-the-box 评测 |
| S21 | arXiv:2402.08191 — The Colosseum | 仿真泛化 benchmark |
| S22 | arXiv:2503.24278 — AutoEval | 真实世界自主评测 |

---

## 二、同期代表性模型事实卡（5 个）

> 每条 2–4 行，只做全景感知，不深入技术细节。

### 1. GR00T N1（NVIDIA，2025 年 3 月）
- **发布机构/时间**：NVIDIA，GTC 2025（2025-03-17 发布）["https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots"]
- **定位一句话**：面向人形机器人的开放 VLA 基础模型，号称"世界首个开放的通用人形机器人基础模型"["https://nvidianews.nvidia.com/_gallery/download_pdf/67d9c5383d6332d0d0bc9af4/"]
- **动作表征类型**：连续动作，由 diffusion transformer（DiT）动作头实时生成，与 System 2 VLM 模块端到端联合训练["https://arxiv.org/pdf/2503.14734v2"]
- **独特贡献**：双系统架构（System 2 视觉-语言推理 + System 1 高频运动控制），训练数据包含第一视角人类视频、真机/仿真轨迹与合成数据["https://arxiv.org/pdf/2503.14734v2"]
- **局限**：主打人形本体（Fourier GR-1、1X 等），真机家庭任务覆盖面有限；后续很快推出 N1.5/N1.6 迭代说明首版仍在快速演进["https://research.nvidia.com/labs/gear/gr00t-n1_5/?ref=vastkind.com"]

### 2. RoboMamba（2024）
- **发布机构/时间**：arXiv 2024-06（北京大学等，作者列表见 Semantic Scholar）["https://www.semanticscholar.org/paper/RoboMamba%3A-Multimodal-State-Space-Model-for-Robot-Liu-Liu/f52c5f1ec94e8a2bf27247bcde7893572c7d53d1/figure/2"]
- **定位一句话**：用 Mamba 状态空间模型（SSM）替代 Transformer backbone 的高效 VLA，同时做机器人推理与操作["https://ar5iv.labs.arxiv.org/html/2406.04339"]
- **动作表征类型**：连续 SE(3) 末端位姿回归（6-DoF 位置+旋转，夹爪任务加 1 维），由冻结主干上接的小型 MLP policy head 直接回归，**非离散 token、非 diffusion/flow**["https://ar5iv.labs.arxiv.org/html/2406.04339"]
- **独特贡献**：3.2B 参数；仅微调 0.1% 参数（3.7M policy head）、数十分钟即可获得操作能力；自称推理速度比同期 VLA 快约 3 倍["https://ar5iv.labs.arxiv.org/html/2406.04339"]
- **局限**：动作头是逐步 SE(3) 回归，未采用 action chunking 等生成式动作建模；主要在 SAPIEN 仿真与 Franka 单臂上验证，泛化证据有限["https://ar5iv.labs.arxiv.org/html/2406.04339"]

### 3. CrossFormer（UC Berkeley + CMU，CoRL 2024 Oral）
- **发布机构/时间**：Ria Doshi、Homer Walke、Oier Mees、Sudeep Dasari、Sergey Levine，UC Berkeley + CMU，CoRL 2024（Top 4% Oral）["https://crossformer-model.github.io/"]
- **定位一句话**：单一 transformer 策略跨 30 种 embodiment 做操作/导航/locomotion/航空，无需观测与动作空间对齐["https://crossformer-model.github.io/"]
- **动作表征类型**：连续动作；decoder-only transformer 共享 backbone，末端按 embodiment 类别接独立 action heads（如双臂 50Hz、四足 12-DoF）["https://crossformer-model.github.io/"]
- **独特贡献**：在 900K trajectories、30 种机器人上训练；首次证明跨具身策略可同时控制单臂、双臂、轮式导航、四足低层运动；比此前跨具身工作在复杂导航/操作上平均成功率高约 3 倍["https://crossformer-model.github.io/"]
- **局限**：语言条件较弱，更接近跨具身模仿学习而非完整 VLA；动作头仍按 embodiment 类别分开，并未真正统一动作空间["https://crossformer-model.github.io/"]
- **【标题勘误】**：任务书给出的标题 *"CrossFormer: A Versatile Vision-Language-Action Model with Arbitrary Modality Combination"* 经检索**未查证到对应论文**；CrossFormer 项目页实际标题为 *"Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation"*（arXiv:2408.11812）["https://crossformer-model.github.io/"]

### 4. Gemini Robotics（Google DeepMind，2025 年 3 月首发）
- **发布机构/时间**：Google DeepMind，2025-03-12 首发 Gemini Robotics + Gemini Robotics-ER；2025-06-24 推出 On-Device 端侧版；2025-09-25 推出 1.5 版["https://goo.gle/Gemini-2-robotics?trk=news_storyline_guest_reshare-text","https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/","https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/"]
- **定位一句话**：基于 Gemini 2.0、把"动作"作为新输出模态的旗舰闭源 VLA，配套 ER（Embodied Reasoning）模块做空间理解与规划["https://goo.gle/Gemini-2-robotics?trk=news_storyline_guest_reshare-text"]
- **动作表征类型**：连续 motor commands（具体动作头结构未公开）；1.5 版在动作前显式输出推理过程["https://arxiv.org/pdf/2510.03342"]
- **独特贡献**：把 Gemini 2.0 的互联网级多模态推理直接接入机器人控制；ER↔VLA 双层栈支持跨 embodiment 技能迁移；On-Device 版仅需约 50 条演示即可微调（媒体报道）["https://www.infoq.com/news/2025/07/google-gemini-robotics/"]
- **局限**：闭源、仅 API/合作访问；训练数据与内部评测细节不透明；外部难以复现["https://www.infoq.com/news/2025/07/google-gemini-robotics/"]

### 5. RDT-1B（清华大学，2024 年 10 月）
- **发布机构/时间**：清华（Songming Liu、Hang Su、Jun Zhu 等），arXiv:2410.07864，2024-10-10 提交["https://www.alphaxiv.org/audio/2410.07864v1","https://git.durrantlab.pitt.edu/thu-ml/RoboticsDiffusionTransformer"]
- **定位一句话**：首个面向双臂操作的 diffusion 基础模型，用 Diffusion Transformer 作为可扩展 backbone["https://arxiv.org/pdf/2410.07864.pdf"]
- **动作表征类型**：**diffusion** 生成连续双臂动作；DiT backbone（约 1.2B 参数，28 层）["https://dl.acm.org/doi/pdf/10.1145/3732945.3733004"]
- **独特贡献**：把 DiT 引入机器人操作以处理多模态输入异构性；专注双臂；开源权重与代码["https://arxiv.org/pdf/2410.07864.pdf"]
- **局限**：双臂任务特定；diffusion 需多步去噪，推理延迟高于自回归/flow 方法；真机验证规模有限["https://arxiv.org/pdf/2410.07864.pdf"]

---

## 三、评估基准与评测体系演进

### 3.1 基准时间线

```
2023-06  LIBERO（UT Austin, NeurIPS 2023）         ← 终身学习/知识迁移仿真基准
2023-10  Open X-Embodiment / RT-X（Google+高校联盟） ← 跨具身数据集+评测协议
2024-02  The Colosseum（2024）                      ← RLBench 上的泛化扰动基准
2024-05  SIMPLER（Stanford/Berkeley/Google, CoRL24）← 真实机器人设置的仿真复现
2024 年  RT-Bench 体系（Google，论文编号【未查证】）  ← 真机 12+17 tasks 评测套件
2024-10  π0 评测（PI 自报）                          ← 5 个真机任务横向对比
2025-03  π0.5 评测（PI 官方博客）                    ← 开放世界/OOD 物体评测
2025-03  AutoEval（Berkeley, CoRL 2025）             ← 真机自主评测
2026-04  π0.7 评测（PI 官方博客）                    ← 与自家 RL specialist 对比，无外部 benchmark
```

### 3.2 每个基准：测什么 / 为什么重要 / 被谁使用

#### (1) LIBERO（2023）
- **测什么**：终身机器人操作学习中的知识迁移；4 个 task suite（Goal / Spatial / Object / Long），每个 10 个连续到达的任务；6-DoF 机械臂+平行夹爪桌面环境["https://arxiv.org/pdf/2306.03310v2.pdf"]
- **为什么重要**：VLA 时代事实上的**仿真操作标准基准**——任务族设计显式区分"目标迁移/空间迁移/物体迁移/长时程迁移"四类泛化维度，评分协议统一，便于横向比较["https://rpl.cs.utexas.edu/publications/2023/12/10/liu-neurips23-libero/"]
- **被谁使用**：OpenVLA、π0/π0-FAST、GR00T 系列、后续几乎所有开源 VLA 论文都报告 LIBERO-Spatial/Object/Goal/Long 分数["https://vnrobo.com/en/blog/vla-lerobot-13-pi0fast-training"]

#### (2) Open X-Embodiment / RT-X（2023）
- **测什么**：跨具身共训是否带来正向迁移；22 种机器人、527 项技能、160K+ 轨迹；6 种机器人上共 3600 次评测试验["https://www.robot-learning.ml/2023/files/paper18.pdf","https://www.roboticscenter.ai/blog/robot-co-training-cross-embodiment"]
- **为什么重要**：建立了 VLA 时代第一个**跨具身共享数据集 + 共享评测协议**；结论是 RT-1-X 比各机器人专属 baseline 平均成功率高约 50%，RT-2-X 在涌现技能评测上比 RT-2 高约 3 倍["https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types","https://robotics-transformer-x.github.io/?ref=nextomoro.com"]
- **被谁使用**：后续跨具身工作（CrossFormer、OpenVLA、GR00T 数据混合）均以此为参考坐标系["https://arxiv.org/pdf/2310.08864v6"]

#### (3) The Colosseum（2024）
- **测什么**：在 RLBench 上施加系统扰动（视觉、动作、物体等），测试操作策略的泛化鲁棒性["https://arxiv.org/pdf/2402.08191?ref=https%3A%2F%2Fgithubhelp.com"]
- **为什么重要**：把"单一任务成功率"细化为**沿泛化维度的扰动响应曲线**，是 VLA 早期仿真泛化评测的代表
- **被谁使用**：早期 VLA/通用操作模型论文常引用其扰动协议作为仿真泛化评测参考["https://arxiv.org/pdf/2402.08191?ref=https%3A%2F%2Fgithubhelp.com"]

#### (4) SIMPLER（2024, CoRL）
- **测什么**：在仿真中复现 Google Robot 与 WidowX BridgeV2 两类真实机器人评测设置，验证仿真评测与真机性能的相关性["https://arxiv.org/html/2405.05941"]
- **为什么重要**：证明**仿真-真实配对评测**可以强相关，且能复现真机上对分布偏移的敏感性；为降低真机评测成本提供工作流["https://simpler-env.github.io/"]
- **被谁使用**：AutoEval、后续通用机器人策略评测论文作为 baseline 仿真环境["https://arxiv.org/html/2503.24278v1/"]

#### (5) RT-Bench 体系（Google, 2024）
- **测什么**：真实机器人上的通用策略评测。OpenVLA 论文可查证部分包括——Google Robot 上 12 个任务（5 个 in-distribution + 7 个 OOD，每个 5 rollout，共 60 rollout）；BridgeData V2/WidowX 上 17 个任务，覆盖 visual / motion / physical / semantic generalization 与 language grounding 五类泛化维度，每任务 10 rollout，含部分奖励（如叠杯半完成给 0.5）["https://arxiv.org/html/2406.09246"]
- **为什么重要**：VLA 时代**真机评测事实标准之一**；把"成功率"拆解到泛化类别上，避免单数字掩盖短板
- **被谁使用**：OpenVLA、RT-1-X/RT-2-X、Octo 等同期通才策略在同一套 Google Robot / Bridge 任务上横向对比["https://arxiv.org/html/2406.09246"]
- **【未查证】**：该评测套件的独立论文编号、项目页 rt-bench.github.io 现无法访问；上表中任务数/rollout 数引自 OpenVLA 论文附录，未独立找到 RT-Bench 原始论文

#### (6) Physical Intelligence π0 / π0.5 / π0.7 官方评测方式
- **π0（2024-10, arXiv:2410.24164）**：在 5 个真机任务上评测——叠衬衫、收拾桌面（easy/hard）、装杂货袋、拿吐司；与 OpenVLA、Octo 在相同训练步数下对比；报告 out-of-box 成功率与语言跟随微调结果["https://arxiv.org/html/2410.24164v4","https://blog.phospho.ai/understanding-p0-by-physical-intelligence-a-vision-language-action-flow-model-for-general-robot-control/"]
- **π0.5（2025-04, pi.website/blog/pi05）**：评测两类条件——①完整清洁任务（把碗碟放进水槽、清理卧室地板物品）；②OOD 评测（按 prompt 把指定物体放进抽屉）；指标为 subtask 平均成功率与跟随率；强调在**完全没见过的家庭环境**中测试["https://www.pi.website/blog/pi05"]
- **π0.7（2026-04, pi.website/blog/pi07）**：与自家 RL 微调的 specialist 模型（π*0.6）在叠衣、咖啡制作、盒子组装等任务上对比；报告 throughput（单位时间完成任务数）而非仅成功率；展示组合泛化（从未训练过的新厨房电器、新机器人平台）["https://www.pi.website/blog/pi07","https://arxiv.org/html/2604.15483"]
- **关键自陈**：PI 官方与多家媒体均指出——**外部通用机器人策略的标准化 benchmark 尚不存在**，π0.7 数字为自报、与自家前作对比，无法外部独立验证["https://awesomeagents.ai/news/physical-intelligence-pi07-generalist-robot/","https://liveinthefuture.org/stories/physical-intelligence-pi07-robot-generalization-data-gap"]

#### (7) AutoEval（Berkeley, 2025, 选录）
- **测什么**：在真实世界**自主**评估通用机器人策略——用视觉模型自动判定任务成败，减少人工评分["https://arxiv.org/html/2503.24278v1/"]
- **为什么重要**：代表 VLA 评测从"人工数 rollout"走向"自动化、可大规模并行"的趋势

---

## 四、关键观察：VLA 评测从"单任务成功率"走向了什么

1. **从单一成功率 → 泛化维度拆解**。LIBERO 把任务拆成 Goal/Spatial/Object/Long 四个迁移维度["https://arxiv.org/pdf/2306.03310v2.pdf"]；OpenVLA 真机评测把 17 个任务拆成 visual/motion/physical/semantic generalization + language grounding 五类["https://arxiv.org/html/2406.09246"]。评测不再问"能不能做"，而是问"在哪个泛化轴上失败"。

2. **从纯仿真 → 仿真/真机配对并验证相关性**。SIMPLER 的核心贡献不是新任务，而是**证明仿真评测与真机性能强相关**["https://simpler-env.github.io/"]；Open X-Embodiment 建立了跨真机评测协议["https://www.deepmind.com/blog/scaling-up-learning-across-many-different-robot-types"]。VLA 论文普遍同时报告 LIBERO（仿真）+ Google Robot/Bridge（真机）两套数字。

3. **从单任务 benchmark → 长时程/开放世界/灵巧操作**。π0.5 主动把评测搬到**没见过的家庭环境**里做清洁与整理["https://www.pi.website/blog/pi05"]；π0.7 评测组合泛化（新电器、新本体）["https://www.pi.website/blog/pi07"]；指标从成功率扩展到 throughput、subtask 部分得分、语言跟随率。

4. **从他评 → 自报与"评测真空"并存**。PI 官方承认外部标准化 benchmark 缺位，π0.7 只能与自家 RL specialist 对比["https://liveinthefuture.org/stories/physical-intelligence-pi07-robot-generalization-data-gap"]；与此同时 AutoEval、ArmnetBench、VLA-REPLICA、ManipArena 等 2025–2026 年新工作试图用**低成本真机农场 + 自动判定**重建可比基准["https://arxiv.org/html/2503.24278v1/","https://arxiv.org/html/2605.20774","https://arxiv.org/html/2603.28545v2"]。

5. **一句话趋势**：VLA 评测正在从"仿真单任务 SR"向"**真机 + 仿真配对、沿泛化轴分层、长时程/开放世界、自动判定、跨本体**"演进；但截至 π0.7 发布时，业界仍未出现一个被普遍接受的"机器人界 ImageNet/MMLU"——这是当前 VLA 报告横向对比最大的方法论短板。
