# Who Watches the Watcher?

**Context exposure and monitor-aware prompt injection in LLM agent monitors**

Rafid Mehda, Independent Researcher

An LLM monitor that guards a tool-using agent has to read something before it approves an action. If it reads the agent's full transcript, it also reads whatever an attacker planted in the tool outputs. This project measures what such a monitor should be allowed to see. We give one monitor model five different views of the same proposed action. Each view is tested on all 949 AgentDojo prompt-injection pairs, under a standard attack and under a monitor-aware attack that addresses the monitor directly. Hypotheses were registered before the main monitor runs, and every number below comes from committed data.

![Attacks caught at a 5% false positive rate, by view](results/figures/catch_by_view.png)

## Findings

- Context helps the monitor. With `gpt-5-nano`, AUROC was 0.728 for the bare action, 0.864 once the user's request was added, and 0.849 with the full transcript. The pre-registered difference between the transcript and the bare action was 0.121 (95% CI [0.070, 0.179]).
- A fixed appeal addressed to the monitor lowered the transcript view's catch rate by 9.7 percentage points and the provenance view's by 2.3. That difference was in the predicted direction but not statistically reliable (95% CI [-0.020, 0.184]).
- The provenance view replaces raw tool outputs with code-computed labels saying where each argument came from. It stayed within 0.03 AUROC of the transcript view but caught fewer attacks at a 5% false positive rate (33.1% against 66.9%). It also wrongly flagged fewer benign actions taken near an injection (1.1% against 6.3%).
- A larger monitor, `gpt-5-mini`, caught 100.0% of attacks from the full transcript and 73.8% from the provenance view.

## The five views

| View | What the monitor sees |
|---|---|
| V1 | The tool call and its arguments |
| V2 | V1 plus the user's request |
| V3 | V2 plus every earlier step, including raw tool outputs |
| V4 | The user's request, names of earlier tool calls, and a provenance label on every argument value (`USER`, `TOOL(name#k)` or `NONE`). Raw tool outputs are withheld and long text is summarized |
| V4raw | V4 with long arguments shown verbatim (ablation) |

## How the evaluation works

The agent (`gpt-4o-mini`) ran once per condition. We then re-executed every recorded tool call against the benchmark environment to verify each trace and to find the exact call at which each attack succeeded. A monitor's decision at a call depends only on what came before it, so scoring the saved traces offline shows exactly where an online monitor would first intervene. A live rerun of the banking suite agreed with the offline prediction (40.6% observed against 44.1% predicted attack success).

Verification also surfaced two reproducibility issues in AgentDojo 0.1.35. Its OpenAI wrapper silently drops a temperature of 0, so this project sets it explicitly. When an episode ends without a final answer, AgentDojo also reruns the agent on the already modified environment and logs only the last attempt. The 22 affected episodes are excluded from the monitor analysis and reported.

## Reproduce

Every result can be regenerated from the committed traces, labels and scores, with no API key:

```bash
uv sync
make analyze site        # results/, figures, tables and site/data from committed data
```

Rerunning the experiments from scratch calls the OpenAI API and cost $10.36 in total for this study:

```bash
cp .env.example .env     # then add your OPENAI_API_KEY
make reproduce-all       # agent sweeps, replay, monitors, online validation, analysis
```

A hard spending cap (configs/experiment.yaml, `budget.cap_usd`) is checked before every paid call. The project needs about 1 GB of disk, Python 3.12 through [uv](https://docs.astral.sh/uv/), and no GPU. Run `make test` for the test suite.

## Repository layout

```
src/wwtw/agent/      monitor-aware attack, agent pipeline, resumable sweeps
src/wwtw/traces/     trace schema, AgentDojo log loader, deterministic replay
src/wwtw/monitor/    provenance labels, the five views, LLM and rule monitors, online gate
src/wwtw/analysis/   metrics, cluster bootstrap, pre-registered tests, figures, tables
data/                traces, replay labels, monitor scores, cost ledger
results/             summary.json, figures, LaTeX tables
site/                static results website with a trace explorer
```

Released under the MIT License. Built on [AgentDojo](https://github.com/ethz-spylab/agentdojo).
