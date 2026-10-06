# SPF 模块 1：二维航点定位

本目录是在当前 `Qwen3-VL-8B-Instruct` baseline 上增加的第一个模块，
实验名为 **Qwen + SPF waypoint**。目前已实现接口和离线验证，导航收益尚未实测。

来源是 [See, Point, Fly](https://github.com/Hu-chih-yao/see-point-fly)，
对应 [论文第 3.1、3.3 节](https://arxiv.org/html/2509.22653v1#S3)。
所选机制是：VLM 在当前图像中预测二维航点，通过相机几何转换成运动方向。
这是适配后的模块实验，不是完整 SPF 复现。

上游仓库的 [LICENSE](https://github.com/Hu-chih-yao/see-point-fly/blob/main/LICENSE)
是限制性许可。本目录依据公开论文独立编写代码和提示词，没有复制或引入上游源码。
来源核对日期：2026-09-11；论文版本：arXiv:2509.22653v1 / CoRL 2025。

## 本次增加的行为

每步先调用现有 `mcts.qwen_vln.QwenVLNBackend.policy()`。
如果它选择 STOP，就直接停止。否则使用**同一个已加载的 Qwen 模型**，
额外调用一次二维航点预测，输入仍为完整指令、相同的最近图像和已执行动作。

模型返回当前图像上的 `[x, y]`，坐标统一归一化到 `[0, 1000]`。
几何控制器使用实际图像宽高、场景相机水平 FOV 和俯仰角，计算 forward/left/up 射线：

- 偏左 / 偏右超过 15°：执行 `TURN_LEFT_30` / `TURN_RIGHT_30`。
- 水平对齐后，仰角 / 俯角超过 15°：执行 `UP_3` / `DOWN_3`。
- 方向对齐后：使用基线在 `FORWARD_3/6/9` 中概率最高的动作。
- 航点不可见或 JSON 不合法：执行这一步原本的基线动作，并记录原因。

每次只执行一个原有动作，再获取新图像重新定位。SPF 本身不产生 STOP。
GPU、推理或几何配置错误会直接报错，不会被伪装成航点回退。

这里选择的是“航点决定方向”的增量实验；条件化的前进步长选择也是适配约定。
原论文中的预测行进距离、自适应步长缩放、障碍物框和连续速度控制未加入这一版。
同时没有叠加 MCTS、`to_do_list` 指令分解或新的 STOP 模块。

## 文件

| 文件 | 作用 |
|---|---|
| `waypoint.py` | 输出协议、提示词、相机投影与动作映射；仅依赖标准库 |
| `policy.py` | 可开关的基线封装、STOP 保留和回退记录 |
| `qwen_backend.py` | 复用基线的模型和 processor，执行航点生成 |
| `evaluate.py` | `base` / `spf` 共用的 AirSim 评测、配置指纹和 JSONL |
| `run_eval.sh` | 使用项目已有 Python / Qwen runtime 的启动脚本 |
| `test_spf.py` | 协议、坐标方向、禁用行为和评测指标测试 |

适配代码全部位于 `spf/`，现有基线、MCTS 和 `to_do_list/` 未作修改。

## 运行

从项目根目录执行。默认数据为 AirSim-23 的 RA50 固定样本集，
最大 40 步、最近 3 帧，模型为本地 Qwen3-VL-8B-Instruct。
第一版评测入口仅支持 AirSim；相机 FOV 从对应场景的
`front_custom` RGB 配置读取。它要求相机安装 yaw/roll 为零。

先做不加载模型、不启动仿真的配置检查：

```bash
bash spf/run_eval.sh --mode spf --dry-run
```

GPU 可用后，先跑第一条指令验证链路：

```bash
bash spf/run_eval.sh --mode spf --max-samples 1
```

正式配对对比，两个命令使用相同配置运行 50 条：

```bash
bash spf/run_eval.sh --mode base --max-samples 50
bash spf/run_eval.sh --mode spf --max-samples 50
```

选择模型 / 渲染 GPU：

```bash
OPENFLY_QWEN_GPU_INDEX=1 OPENFLY_AIRSIM_GPU_INDEX=0 bash spf/run_eval.sh --mode spf
```

可选参数有 `--eval-info`、`--start-sample`、`--max-step`、`--history-images`、
`--model-path`、`--attention`、`--device`、`--max-new-tokens`、`--seed`、`--results`。
`--max-samples 0` 表示运行起始索引后的所有样本。
两组必须使用同一版本的代码和相同评测参数。

结果默认写到 `spf/outputs/<mode>_<timestamp>_<pid>.jsonl`。
指定 `--results` 时目标文件必须不存在；不覆盖或拼接旧实验。
每条完成的样本都立即写入。中断后的新实验可指定 `--start-sample`，
汇总前须检查不同文件的配置指纹及样本索引，避免遗漏或重复计数。

也可以只验证一张图像的真实模型推理：

```bash
bash spf/run_eval.sh --mode spf --max-samples 1 --image test/cur_img.jpg
```

该命令不启动仿真，只把第一条指令与指定图像送给模型，并向标准输出打印决定。
`test/cur_img.jpg` 未经核实是该条任务的真实观测，因此这种检查**不能作为导航评测**。
如果基线当步选择 STOP，日志中将显示 `baseline_stop`，航点生成也不会被调用。

## 如何判断这个模块是否有效

主对比应使用这里同一入口产生的新 `base` 和 `spf` 结果。
项目旧的 RA50 基线成绩（SR 10%）来自另一版动作提示和 JSON 生成策略，
不宜直接用来计算这一模块的增益。

`base` 模式直接透传当前基线动作，保留完整指令、历史帧、动作集合和推理实现。
两组使用同一环境接口和指标逻辑，与 `mcts/base_qwen.py` 对齐：

- NE：最终位置到目标位置的欧氏距离，单位米。
- SR：最终 NE 严格小于 20 米；不额外要求 STOP。日志单独记录 `stopped`。
- OSR：至少一个执行动作后的位置在目标 20 米内，或最终 SR 为 1。
- SPL：成功时以起终点直线距离除以实际路径长度与直线距离的较大值。

JSONL 中 SR / OSR / SPL 是 `0..1`，转百分数时乘 100。
这里的 SPL 沿用项目的直线距离近似，不能当作经过障碍物规划的最短路径。
目标坐标只参与评分，不进入基线或航点模型输入。

除 NE / SR / OSR / SPL，还应统计：

- `decision.source`：实际用上航点的比例、基线 STOP 和回退次数。
- `decision.fallback_reason`：解析错误与不可见航点的占比。
- `decision.action` 与 `decision.baseline.action`：航点实际改变了多少动作。
- `decision_seconds`：增加一次生成调用后的推理耗时。

日志保存原始模型输出、二维航点、投影方向、原始基线分布和实际动作，
不会把几何控制器的决定描述为模型给出的动作概率。
首条配置记录保存数据 / 源码 / 相机设置 SHA256、模型路径和参数；
实际运行还会记录基础提示版本及 torch / transformers 版本。

SPF 的论文结果不代表它在 Qwen3-VL-8B + OpenFly 上一定有效。
第一版控制器采用简单阈值量化，可能出现反复转向、持续升降或目标不可见时回退。
STOP 仍继承基线的误停问题；单目航点没有真实距离信息，也没有新增碰撞检测。
应先检查单条轨迹，再报告完整 RA50 配对结果。

## 已完成验证

```bash
/home/liusongbo/miniconda3/envs/openfly/bin/python -m unittest spf.test_spf -v
```

测试覆盖左右/上下动作与项目真实运动函数的符号一致性、相机俯仰和宽高比、
非法输出与不可见航点回退、基线 STOP、模块关闭后的动作等价，以及
终步到达 / 经过目标后过冲等指标边界。

已通过 18 项离线测试、两种模式的 RA50 配置检查、shell 语法检查。
另使用本地真实 Qwen processor 和 `test/cur_img.jpg` 完成了提示词与图像预处理检查：
1920×1080 图像成功编码，输入 token 形状为 `[1, 2360]`；这不包含模型生成。

当前工作环境检测到 `torch.cuda.is_available() == False`，且无法访问 NVIDIA 驱动。
尚未完成真实 Qwen 航点生成或 AirSim 闭环评测，暂无本模块的 SR / SPL 成绩。
