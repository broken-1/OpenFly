# Instruction Progress Tracker

这是一个独立于 MCTS 的想法：把一条长 VLN 指令先拆成有序子目标，再让 Qwen 在执行过程中判断当前子目标的完成度。它不改变动作策略，也不进入 MCTS 搜索树；后续可以作为一个上层任务状态模块，为任意动作控制器提供当前阶段上下文。

## 核心闭环

1. 任务开始时，Qwen 只调用一次，把原始 instruction 拆成 `1, 2, 3, ... N` 个有顺序的子目标。
2. 最后一个子目标必须是最终目标，完成条件固定为“导航到最终目标 20 米内并停下”。这样最终成功判定不会依赖模型自由发挥。
3. 每次获得新的真实观测后，只评估当前 active subgoal。模型不能直接把当前步骤改成第 `k+2` 步。
4. 当当前子目标返回 `COMPLETED` 且置信度达到阈值，状态机才推进到下一个子目标，并在下一次 prompt 中明确写成 `Currently executing subgoal 2 of N`。
5. `IN_PROGRESS` 保持当前子目标；`BLOCKED` 记录阻塞证据，暂不自动跳步。重规划是后续扩展，不在第一版中隐式发生。

初版协议和状态机见 [instruction_progress.py](instruction_progress.py)。它通过注入 `generate(prompt, images)` 接入 Qwen，因此不需要导入现有 `mcts/`，也不要求现在加载模型或启动仿真。

## 为什么不让模型直接预测“下一步编号”

模型可以判断当前子目标是否完成，但“跳过步骤”会破坏 instruction 的顺序约束。状态机保留唯一的 `active_index`，只允许完成当前步骤后前进一格。模型输出的是证据、完成度和置信度，顺序推进由代码负责。

## 第一版 Prompt 协议

规划器输出：

```json
{
  "final_target": "the final target",
  "subgoals": [
    {
      "id": 1,
      "description": "first ordered subgoal",
      "completion_criterion": "observable completion evidence"
    }
  ]
}
```

跟踪器输出：

```json
{
  "subgoal_id": 1,
  "status": "IN_PROGRESS",
  "progress": 0.45,
  "confidence": 0.82,
  "evidence": "the landmark is visible but the required alignment is not complete"
}
```

`progress` 是当前子目标的完成度，不是整条任务的成功概率；`confidence` 只用于是否允许状态机推进。默认完成置信度阈值是 `0.75`，可在实验中调节。

## 建议的研究问题

- 分解是否真的提升长指令导航，而不是只增加模型调用次数？
- 子目标完成判断比单次 Arrival/STOP 判断是否更稳定？
- 每步跟踪、每两步跟踪、只在转向或阶段变化后跟踪，哪种成本/效果最好？
- 子目标完成度能否作为可解释的中间指标，并预测最终 SR、NE 和 OSR？
- 当模型输出 `BLOCKED` 时，是局部重新观察、回退，还是重新规划，哪种策略最合适？

## 待办

- [ ] 用正式 Qwen3-VL 推理后端实现 `generate(prompt, images)` 适配器。
- [ ] 在同一批约 10 条样本上标注人工子目标和完成时刻，评估自动分解质量。
- [ ] 比较“每步跟踪”和“事件触发跟踪”的调用成本与完成判断准确率。
- [ ] 增加人工/几何证据接口，例如最终距离 `< 20m` 时校验最后一步，而不是只依赖视觉语言模型。
- [ ] 设计 `BLOCKED` 的恢复策略和显式 `REPLAN` 版本。
- [ ] 再考虑把状态上下文接到 Base Policy；这一步仍然不等于接入 MCTS。

## RA50 sample 0 验证脚本

`run_ra50_first.py` 只读取 RA50 JSON 的第 1 条 instruction，执行一次真实 Qwen Planner 和一次 Tracker。它不会启动 AirSim，也不会执行动作。由于 RA50 JSON 本身不含相机帧，Tracker 默认使用 `test/cur_img.jpg`，结果中会标记 `image_is_sample_zero_frame: false`；拿到 sample 0 的真实起始帧后，可通过 `OPENFLY_TODO_IMAGE=/path/to/frame.jpg` 替换。
