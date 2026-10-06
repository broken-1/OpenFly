# Experiment and version management

- Work on a named branch and commit source/config changes before starting an experiment. Preserve existing user changes; never rewrite history or silently change a running experiment.
- Before comparing to an existing result, inspect that result's logs and decision protocol. The filename `base_qwen.py` is not proof of equivalent behavior.
- Distinguish `generated_action_reason_json`, `action_token_logits_argmax`, and `mcts`. Changing prompts, action parsing, STOP rules, preprocessing, or step limits creates a different experiment.
- Use an explicit run configuration. Always record the model/revision, precision, inference backend and versions, thinking mode, prompt version/hash, max steps, image history, dataset paths/hashes, seed, renderer/GPU, and launch arguments. Never rely on a Python default being the effective setting: shell launchers can override it.
- Save a Git commit, effective config and clean/dirty status to each run manifest before starting. Use `scripts/record_experiment.py`. It rejects tracked or untracked source changes; commit them first.
- Never attach a historical result to a guessed commit. If source or dataset provenance is missing, record it as unknown. Current file hashes do not prove what was run in July.
- Keep model weights, simulator binaries, images and bulky outputs outside Git. Commit source, configurations, small aggregate scores and experiment registry entries. Use a unique output directory for every run.
- Report completed sample counts and whether comparisons use identical tasks and protocols. Partial or changed-policy trials are not replacement baseline scores.
- Before completing code work, run appropriate validation, commit it on the working branch, and report the commit and experiment identifier. Sync to the configured GitHub repository when authorized; keep the main branch unchanged unless explicitly asked to merge.
