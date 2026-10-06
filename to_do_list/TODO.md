# Implementation TODO

## Done in the standalone prototype

- [x] Ordered subgoal data model
- [x] Planner JSON prompt and parser
- [x] Tracker JSON prompt and parser
- [x] Ordered state transitions with no step skipping
- [x] Final target normalization to the 20-meter completion criterion
- [x] Injectable model-call boundary for Qwen

## Next experiments

- [ ] Define the exact visual evidence for each subgoal type: landmark reached, heading aligned, passage completed, and final target reached.
- [ ] Measure decomposition quality separately from navigation quality.
- [ ] Compare completion confidence thresholds `0.60 / 0.75 / 0.90`.
- [ ] Compare tracker cadence: every observation versus event-triggered observation.
- [ ] Add a small human-labeled replay evaluator before any full AirSim run.
- [ ] Add explicit replan records instead of silently replacing the original plan.
