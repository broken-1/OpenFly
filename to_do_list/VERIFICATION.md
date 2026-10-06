# RA50 sample 0 standalone verification

验证脚本：`python to_do_list/run_ra50_first.py`

结果文件：[ra50_sample0_verification_final.json](outputs/ra50_sample0_verification_final.json)

## 验证内容

- 输入：`configs/seen_airsim_23_random50_seed20260609.json` 的第 1 条（`sample_idx=0`）。
- Planner：真实加载 Qwen3-VL-8B-Instruct，生成并解析有序子目标。
- Normalizer：去除重复的最终到达步骤，确保只保留一个最终目标；前置步骤不使用最终 `20m + STOP` criterion。
- Tracker：真实调用 Qwen 对当前子目标进行状态判断。
- State machine：成功接收 Tracker JSON，并保持 `active_subgoal=1`。
- MCTS/AirSim：未导入、未启动、未执行动作。

## 最终计划

1. 直行前往蓝白色大型金属框架吊车。
2. 稍微左转并继续直行。
3. 到达灰色、带水平格栅和垂直网格的现代建筑立面；进入 20 米内并停止。

## Tracker 结果

Tracker 返回：

```json
{
  "subgoal_id": 1,
  "status": "BLOCKED",
  "progress": 0.0,
  "confidence": 1.0
}
```

这是因为当前用于烟雾测试的 `test/cur_img.jpg` 是一张空旷地平线图，不是 RA50 sample 0 的真实起始帧。状态机按设计保持：

```text
active_index = 0
Step 1 = BLOCKED
Step 2 = PENDING
Step 3 = PENDING
```

因此，本次已验证“Planner → 规范化 → Tracker → 状态机”的程序链路；还没有验证 sample 0 起始视觉帧上的判断质量。拿到 sample 0 的真实相机帧后，设置 `OPENFLY_TODO_IMAGE` 即可复测 Tracker。
