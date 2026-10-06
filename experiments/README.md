# 实验版本登记

`registry.json` 登记现有成绩的来源、配置和可比较性。原始逐步日志仍在本机 `outputs/`，不上传模型或图像。

目前的两份结果：

| 实验 ID | 决策方式 | 最大步数 | 状态 | 源码版本 |
| --- | --- | ---: | --- | --- |
| qwen8b-json-base-20260715 | 生成 action/reason JSON，再解析动作 | 100 | 1392 条完成 | 未归档，不能确认 |
| qwen27b-token-base-20261005 | 0–9 动作 token 概率取最大值 | 100 | 94 条后停止 | 当时未提交；事后源码快照已记录 |

两者策略不同，不能作为仅替换模型的公平比较。目录名和程序名不能替代源码版本。登记时计算的数据集哈希也不能反推历史运行时的数据版本。

## 后续运行流程

1. 在分支上修改代码和配置，检查并提交。
2. 写入明确的 JSON 运行配置，包含 `experiment_id`、`model`、`decision_protocol`、`max_steps`、`history_images`、`datasets`、`runtime`、`prompt`、`seed` 和 `launch_command`。`datasets` 是仓库相对路径列表；runtime/prompt 要包含版本及设置。
3. 运行前保存 manifest：

```bash
python scripts/record_experiment.py --config experiments/my_run.json --output outputs/my_run/run_manifest.json
```

该命令只登记，不启动评测；工作区未提交、已有同名 manifest 或数据文件缺失时会拒绝。

4. 按登记的配置启动评测，结果写入对应独立目录。更改决策方式或步数限制，应新建实验 ID，不能续写原成绩。
5. 完成后把样本数、聚合指标、输出目录和提交号登记到 registry，再提交登记变更。

对齐截图中的旧 8B 成绩，需要恢复 JSON 决策逻辑及旧提示词；当前 token 打分程序不能直接宣称复现旧版。若使用 40 步，应给 8B 和 27B 各建立明确的 40 步实验。
