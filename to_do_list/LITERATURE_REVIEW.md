# 指令分解与进度跟踪文献调研

调研日期：2026-08-18

## 结论

“用大模型把长导航指令分解为有序子目标，在执行过程中跟踪当前子目标，完成后切换到下一子目标，并在最后目标处停止”这一基础思想已经被公开论文覆盖，不能再将其整体作为全新的方法贡献。

与当前方案最接近的工作是 AgriVLN 和 OnFly：

- AgriVLN 已经使用 LLM 将指令一次性分解为 Subtask List，并为每个子任务维护 `pending / doing / done` 状态。VLM 每一步接收子任务列表，只关注当前子任务，同时输出动作和状态变化。
- OnFly 已经在空中 VLN 中用机载 VLM 将长指令一次性分解为子任务队列。Decision Agent 接收当前子任务执行导航，Monitoring Agent 输出 `CONTINUE / STOP / LOST`；当前子任务稳定输出 `STOP` 后，Task Manager 切换到下一子任务，最后一个子任务完成后结束整个 episode。

因此，当前方法不能以“Instruction Decomposition + Progress Tracking”本身作为主要新颖性。更可行的贡献方向是：面向空中 VLN 的**异构子目标类型与类型相关的完成判据**，特别是将视觉地标、动作机动和最终目标分别处理。

## 与当前想法的逐项对比

| 当前想法 | 已有先例 | 重合程度 |
|---|---|---|
| 长指令由 LLM/VLM 一次性拆成 Step 1、2、3… | AgriVLN、OnFly、HSGM、AERIS | 高 |
| 只向动作模型强调当前 Step | AgriVLN、OnFly、AERIS | 高 |
| 完成当前 Step 后更新状态并切换下一 Step | AgriVLN、OnFly、HSGM | 高 |
| 使用视觉模型判断当前子目标是否完成 | AgriVLN、OnFly、HSGM | 高 |
| 最后一步到达最终目标并 STOP | AgriVLN、OnFly、HSGM | 高 |
| 空中 VLN 中的长时进度管理 | OnFly、AERIS | 高 |
| `LANDMARK / MANEUVER / FINAL_TARGET` 类型化子目标 | 本次检索未发现完全相同的明确设计 | 可能有空间 |
| `MANEUVER` 用实际动作/航向变化确定性完成，而非让视觉模型猜测 | 本次检索未发现完全相同的明确设计 | 较有空间 |
| 不同类型使用不同证据与动作空间 | 已有层级控制思想，但当前具体组合未发现直接同款 | 可能有空间 |

## 高度相关工作

### 1. AgriVLN: Vision-and-Language Navigation for Agricultural Robots

- arXiv: [2508.07406](https://arxiv.org/abs/2508.07406)
- 首次提交：2025-08-10
- 当前检索到的状态：arXiv 预印本；arXiv 元数据未注明正式会议或期刊。
- 接近程度：非常高。

其 Subtask List 中的每个子任务包含：

```text
ID
description
start condition
end condition
state: pending / doing / done
```

指令只在 episode 开始时分解一次。每个时刻，VLM 同时接收当前图像和 Subtask List，输出低层动作以及子任务状态变化。论文明确规定：前一子任务完成后，下一子任务才能从 `pending` 进入 `doing`；当前子任务完成后从 `doing` 进入 `done`。这与“Prompt 从正在执行第 1 步切换为正在执行第 2 步”在功能上基本相同。

论文报告加入 STL 后，整体 SR 从 0.33 提高到 0.47。论文结论部分存在另一组 0.31 到 0.42 的数字，可能是稿件版本未统一；引用时应以正式表格和最终版本为准。

### 2. OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency

- arXiv: [2603.10682](https://arxiv.org/abs/2603.10682)
- 首次提交：2026-03-11
- 当前检索到的状态：DBLP 仅收录为 CoRR/arXiv，未发现正式会议或期刊信息。
- 接近程度：最高，而且同属空中 VLN。

OnFly 的 Task Manager 使用机载 VLM 在收到指令时进行一次分解，形成子任务队列。执行阶段包括：

```text
当前子任务
    -> Decision Agent 高频生成飞行目标
    -> Monitoring Agent 低频判断 CONTINUE / STOP / LOST
    -> 当前子任务稳定 STOP 后切换下一子任务
    -> 最后子任务 STOP 后结束任务
```

它还使用“起始帧 + 关键帧 + 最新帧”的 Hybrid Memory 做长时进度监控。这意味着“一次分解、当前目标 Prompt、完成判断、逐步切换、最终停止”的完整闭环在空中 VLN 中已经出现。

### 3. AERIS: Aerial-Edge Role-Driven Intelligence at Runtime via Orchestrated Language-Model Swarm

- arXiv: [2606.30151](https://arxiv.org/abs/2606.30151)
- 首次提交：2026-06-29
- 当前检索到的状态：arXiv comment 明确写着 `Preprint version of the submitted manuscript`，即投稿中的预印本，不能视为已正式录用。
- 接近程度：高，同属 UAV 长指令执行。

AERIS 将指令切分为子目标，利用 Attention–Subgoal Alignment 根据当前状态选择 active subgoal，并在系统消息中标注当前 active instruction step。它与“把当前正在执行第几步写入 Prompt/消息”直接重合，但 active step 的选择更偏向注意力对齐，而不是严格的顺序状态机。

### 4. Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation

- arXiv: [2606.00095](https://arxiv.org/abs/2606.00095)
- 首次提交：2026-05-25
- 当前检索到的状态：arXiv 元数据未注明正式会议或期刊，按预印本处理。
- 场景：室内连续环境 VLN，不是空中 VLN。
- 接近程度：非常高。

HSGM 使用 LLM 将复杂指令分解为有序、可执行、具有明确终止条件的子任务，并维护：

```text
pending / in_progress / done
```

VLM 持续接收 `in_progress` 子任务；输出特殊 `STOP` 后，当前子任务变为 `done`，并激活下一子任务。最终子任务还使用连续两次 STOP 确认。论文报告移除子任务分解后 SR 下降 8.9%。

### 5. Sub-Instruction Aware Vision-and-Language Navigation

- arXiv: [2004.02707](https://arxiv.org/abs/2004.02707)
- 正式发表：EMNLP 2020
- ACL Anthology: [2020.emnlp-main.271](https://aclanthology.org/2020.emnlp-main.271/)

该工作为 R2R 增加子指令及其对应路径标注，并设计 sub-instruction attention 和 shifting modules，使 agent 每个时刻选择并关注一个子指令。它证明“把完整指令切成小段并跟踪当前段”在经典 VLN 中早已有正式发表先例，但它依赖训练数据与细粒度标注，不是现在这种零样本 LLM Planner/Tracker。

### 6. Self-Monitoring Navigation Agent via Auxiliary Progress Estimation

- arXiv: [1901.03035](https://arxiv.org/abs/1901.03035)
- 正式发表：ICLR 2019

该工作明确研究 agent 如何知道已完成哪部分指令、下一部分是什么以及当前整体进度。它没有使用现代 LLM 生成显式子任务列表，但“导航进度监控”本身已经是经典方向。

### 7. The Regretful Agent: Heuristic-Aided Navigation through Progress Estimation

- arXiv: [1903.01602](https://arxiv.org/abs/1903.01602)
- 正式发表：CVPR 2019 Oral

该工作使用学习到的 progress monitor 指导继续前进或回退，说明进度估计和基于进度的纠错也不能单独作为新贡献。

## 新颖性判断

### 不建议声称的创新

- 首次用 LLM/VLM 分解长导航指令。
- 首次维护当前子目标或当前 instruction step。
- 首次在子目标完成后切换下一子目标。
- 首次通过 Prompt 告诉模型当前正在执行第几步。
- 首次在空中 VLN 中做长时进度监控。

这些表述均容易被 AgriVLN、OnFly、AERIS 或 HSGM 直接反驳。

### 建议收敛的核心贡献

建议将方法从笼统的“分解 + 跟踪”收敛为：

> 面向空中 VLN 的类型感知子目标状态机，根据子目标的语义类型选择不同的完成证据、动作约束和状态转移规则。

具体可以定义：

```text
LANDMARK
  完成证据：视觉时序证据，确认已接近或经过指定地标

MANEUVER
  完成证据：已执行动作、累计航向/高度变化、动作持续距离
  不让 VLM 仅凭图像长期猜测“稍微左转”是否完成

FINAL_TARGET
  完成证据：多帧/多视角目标确认和停止置信度
  控制器只在该阶段允许最终 STOP
```

这个方向直接对应 RA50 sample 0 暴露的问题：`Slightly turn left while continuing straight` 是机动指令，却被视觉 Tracker 长期保持为 `IN_PROGRESS`，导致 agent 没有进入最终目标子任务并持续前进过冲。

## 建议的论文问题与实验

主问题不应再是“分解是否有效”，而应改成：

1. 统一使用 VLM 判断所有子目标完成，为什么在空中 VLN 中不可靠？
2. 类型感知的混合完成判据是否减少子目标卡死、误切换和过冲？
3. 更准确的中间状态转换是否最终改善 SR、SPL、NE 和 STOP precision？

建议至少进行以下消融：

```text
Base Qwen：完整指令直接导航
Decomposition-only：只拆分，按固定步数或简单规则切换
VLM Tracker：所有子目标都由 VLM 判断完成
Typed Tracker：LANDMARK / MANEUVER / FINAL_TARGET 使用不同完成判据
Typed Tracker + type-aware action constraints：完整方法
```

除了最终导航指标，还应报告：

- Subgoal Completion Accuracy
- Subgoal Transition Precision / Recall
- Premature Transition Rate
- Stuck Subgoal Rate
- 到达目标附近后的 Overrun Distance
- Final STOP Precision / Recall
- 每条指令完成的子目标比例

这些中间指标能证明提升确实来自进度管理，而不只是走得更久或偶然停止。

## 当前定位建议

当前原型仍然有研究价值，但应将定位改为：

> 不是提出“指令分解 + 进度跟踪”，而是研究空中 VLN 中不同语义子目标为何需要异构完成机制，并构建类型感知的闭环子目标执行框架。

该定位比基础想法更具体，也能由 sample 0 的真实失败案例直接支撑。不过，正式投稿前仍需继续追踪 OnFly、AERIS 和 HSGM 的后续版本及录用状态，因为它们均是 2026 年出现的高相关预印本。
