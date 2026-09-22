# Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

Code and data for the Findings of EMNLP 2026 paper
([arXiv:2609.00949](https://arxiv.org/abs/2609.00949)).

We classify every model emission in a multi-turn tool-calling conversation into
one of four action classes (TOOL_CALL, ASK, REFUSE, CONFIRM). Each BFCL
multi-turn category fixes a gold class: TOOL_CALL for `base` and
`long_context`, ASK for `miss_param`, and REFUSE for `miss_func`. **Gold Action
Recall (GAR)** is the share of cases in which the model emits the gold class.
Paired with the benchmark's state-graded accuracy (Acc), GAR bounds Acc from
above whenever the gold class is required for success. **Acc > GAR** means the
grader passed trajectories whose action class was wrong, and **GAR ≫ Acc**
places the failure in execution. Two inference-time probes, **SRI** (halt seam)
and **CRI** (call seam), test how far that calibration moves.

## Install

Python ≥ 3.10.

```bash
pip install -e .              # classifier, GAR, and the released-data scripts (no dependencies)
pip install -e ".[bfcl]"      # + BFCL multi-turn on API models (bfcl-eval 2026.3.23)
pip install -e ".[vllm]"      # + local checkpoints served with vLLM (Linux + CUDA)
pip install -e ".[tau2]"      # + tau2-bench at the pinned commit (Python ≥ 3.12)
pip install -e ".[judge]"     # + the OpenAI client for the CONFIRM judge
```

bfcl-eval pins vLLM 0.8.5, which predates gpt-oss. For gpt-oss checkpoints,
install a vLLM release with gpt-oss support (e.g. 0.19) after the `vllm` extra.

## BFCL multi-turn

The paper's BFCL v3 multi-turn categories are the four `multi_turn_*`
categories of bfcl-eval 2026.3.23, 200 conversations each.

Local checkpoints run in native function-calling mode, served with vLLM:

```bash
python -m bfcl_calibration.evaluation.bfcl --model-path <hf-checkpoint-dir> --output-dir runs/<name>
```

The driver registers the checkpoint with upstream BFCL's handler for its
model family: by name for Hammer, ToolACE, Watt-Tool, Granite-3, and xLAM-2,
and by `config.json` `model_type` for Qwen3 and Llama-3.1. gpt-oss checkpoints
use this repo's handler (`bfcl_calibration/evaluation/bfcl/_gptoss_handlers.py`),
which queries vLLM through the Responses API. `BFCL_VLLM_EXTRA_ARGS` passes
extra flags to `vllm serve`, and `BFCL_SEED` adds `--seed`. Rerunning the same
command resumes an interrupted run.

API models go through upstream's OpenAI-compatible function-calling handler:

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://<endpoint>/v1
python scripts/run_bfcl_api.py --model-id gpt-5.4-2026-03-05 --reasoning-effort high --out-dir runs/gpt-5.4
```

The paper's gpt-5.4 row uses `--reasoning-effort high`; the other API rows
leave reasoning effort at the server default. gpt-5 accepts only
`--temperature 1`.

GAR and Acc come from BFCL's own result and score files, so any multi-turn run
works:

```bash
python -m bfcl_calibration.analysis.gar --results runs/gpt-5.4/bfcl_results --scores runs/gpt-5.4/bfcl_scores
```

```
category      gold           n     GAR     Acc  GAR-Acc
base          TOOL_CALL    200    99.0    57.0    +42.0
long_context  TOOL_CALL    200   100.0    56.0    +44.0
miss_func     REFUSE       200    65.5    52.0    +13.5
miss_param    ASK          200    44.0    41.5     +2.5
overall Acc                      51.62
```

The action classifier is `bfcl_calibration/analysis/action_classifier.py`. It
holds the cue lists and the within-emission priority rule.

## Probes

Both probes act inside BFCL's multi-turn loop. Environment variables turn them
on, for local checkpoints and API models alike:

| Probe | Enable | Variants |
|---|---|---|
| SRI (halt seam) | `BFCL_SRI_MODE=sri_only` | `BFCL_SRI_VARIANT`: `v1` (default); ablations `ablation_no_history`, `ablation_no_state`, `ablation_no_recon`; SRI-Gate `decision_routed_v1` (global) and `decision_routed_v2_a` (*miss_param*) |
| CRI (call seam) | `BFCL_CRI_MODE=cri_only` | `BFCL_CRI_VARIANT`: `v1` = CRI-retry (default), `bypass` = CRI-bypass |

```bash
BFCL_SRI_MODE=sri_only python -m bfcl_calibration.evaluation.bfcl --model-path <dir> --output-dir runs/<name>-sri
BFCL_CRI_MODE=cri_only BFCL_CRI_VARIANT=bypass python -m bfcl_calibration.evaluation.bfcl --model-path <dir> \
    --output-dir runs/<name>-cri-bypass --categories multi_turn_miss_func multi_turn_miss_param
```

The paper's CRI runs cover *miss_func* and *miss_param*. Each probe run writes
a JSONL log of every trigger decision to `<output-dir>/sri_event_log.jsonl`
(`BFCL_SRI_LOG_PATH` overrides the path). The prompt templates are in
`bfcl_calibration/evaluation/bfcl/_sri.py` and `_cri.py`.

## τ²-bench

The runs use tau2-bench at commit `a03b7910`. Its task data comes from a
checkout:

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout a03b7910bc968f706306e16017a9cb650caf7af2
export TAU2_DATA_DIR=$PWD/tau2-bench/data
```

**Diagnostic (Tab. 2, Tab. 3).** These runs use tau2's own agent on all tasks
(airline 50, retail 114), one trial each. gpt-5.4-2026-03-05 at low reasoning
effort simulates the user and grades retail's natural-language assertions.
`bfcl_calibration.evaluation.tau2_run` passes its arguments to `tau2 run` and
lets `TAU2_NL_ASSERTIONS_LLM` replace the assertion judge:

```bash
export TAU2_NL_ASSERTIONS_LLM=openai/gpt-5.4-2026-03-05
export TAU2_NL_ASSERTIONS_ARGS_JSON='{"temperature": 0.0, "max_tokens": 2048, "response_format": {"type": "json_object"}, "reasoning_effort": "low"}'
python -m bfcl_calibration.evaluation.tau2_run --domain retail --num-trials 1 \
    --agent-llm <litellm model> --agent-llm-args '{"temperature": 0.0}' \
    --user-llm openai/gpt-5.4-2026-03-05 --user-llm-args '{"temperature": 0.0, "reasoning_effort": "low"}' \
    --save-to tau2_retail_<model>_baseline
```

For gpt-5-family agents, add `"reasoning_effort": "high"` to `--agent-llm-args`.
tau2 writes each run to `$TAU2_DATA_DIR/simulations/<save-to>/results.json`.
From there:

```bash
python scripts/classify_tau2_tasks.py --tau2-root tau2-bench --output-json tau2_tasks.json
python scripts/compute_tau2_calibration_breakdown.py --runs $TAU2_DATA_DIR/simulations \
    --task-classification tau2_tasks.json --tau2-root tau2-bench --output-json tau2_breakdown.json
python scripts/tau2_ask_subset.py --run <model> retail $TAU2_DATA_DIR/simulations/tau2_retail_<model>_baseline/results.json
python scripts/judge_tau2_confirm.py --run <model> retail $TAU2_DATA_DIR/simulations/tau2_retail_<model>_baseline/results.json \
    --out confirm_judgments.jsonl
python scripts/confirm_table.py --judgments confirm_judgments.jsonl
```

The breakdown gives pass@1 and the TOOL_CALL and REFUSE columns of Tab. 2, and
`tau2_ask_subset.py` gives the ASK columns. `judge_tau2_confirm.py` judges every
database write with gpt-5.4 (it reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`),
and `confirm_table.py` turns the judgments into Tab. 3.
`scripts/tau2_reward_basis_audit.py --tau2-root tau2-bench` counts which tasks
use the natural-language-assertion judge.

**SRI-lite (App.).** The launcher installs the SRI-lite agent in place of
tau2's default agent. `SRI_LITE_VARIANT` selects `baseline`, `sri_lite`, or
`ablation_no_state`. In these runs the agent model also simulates the user
(at temperature 0.7) and grades the assertions:

```bash
export TAU2_NL_ASSERTIONS_LLM=<model>
export TAU2_NL_ASSERTIONS_ARGS_JSON='{"temperature": 0.0, "max_tokens": 800, "response_format": {"type": "json_object"}}'
SRI_LITE_VARIANT=sri_lite python -m bfcl_calibration.evaluation.tau2_sri_lite_launcher --domain retail \
    --num-trials 1 --max-steps 40 \
    --agent-llm <model> --agent-llm-args '{"temperature": 0.0, "max_tokens": 4000}' \
    --user-llm <model> --user-llm-args '{"temperature": 0.7, "max_tokens": 4000}' \
    --save-to tau2_retail_<model>_sri_lite
```

For a model served locally with vLLM, add its `api_base` to the three argument
sets. For xLAM-2-8b, which writes tool calls as JSON text, also set
`TAU2_COERCE_JSON_CONTENT_TOOL_CALLS=1`.

## Released data

`data/` holds the annotations the paper releases. Field-level documentation is
in [`data/README.md`](data/README.md).

| File | Contents |
|---|---|
| `audit_labels.jsonl`, `audit_rubric.md` | 200-case mechanism audit and 20-case pilot: case contexts, both raters' labels, adjudication, final labels; the frozen rubric |
| `tau2_confirm_judgments.jsonl` | 1,887 database writes on τ²-bench with the judge's verdict (1,879 judged) |
| `tau2_ask_label_audit.jsonl` | the 41 τ²-bench ASK candidates, each checked against its task text |

## Reproducing the paper

| Result | How |
|---|---|
| Tab. 1 (GAR / Acc on BFCL) | one run per model (local or API), then `python -m bfcl_calibration.analysis.gar` |
| Tab. 2 (τ²-bench per class) | diagnostic runs, then `compute_tau2_calibration_breakdown.py` and `tau2_ask_subset.py` |
| Tab. 3 (confirmation before writes) | `python scripts/confirm_table.py` |
| Tab. 4 (SRI / CRI shifts) | probe runs, then `python -m bfcl_calibration.analysis.gar` |
| Tabs. 15–17 (audit agreement and rates) | `python scripts/audit_stats.py` |

Tab. 3 and Tabs. 15–17 come from the released data alone.

## Citation

```bibtex
@inproceedings{zhao2026calibration,
  title         = {Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling},
  author        = {Zhao, Kangjia and Li, Jiajun and Shen, Haozhan and Chow, Wei and Li, Linfeng and
                   Song, Hang and Kong, Lingdong and Zhi, Chen and Zhao, Tiancheng and Liu, Songhua and
                   Yin, Jianwei},
  booktitle     = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year          = {2026},
  eprint        = {2609.00949},
  archivePrefix = {arXiv}
}
```

## License

Apache 2.0 (see [`LICENSE`](LICENSE)). BFCL and tau2-bench are used as
dependencies under their own licenses.
