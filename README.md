# AgMS-foundation

Code and results for the paper

> **Mathematical Foundations of Agentic Mobility Systems: Human and Machine Agency, Autonomy, and Trustworthy Delegation**
> Jiangbo Yu and Ali Eslami, McGill University (2026). Submitted to *Transportation Research Part B: Methodological*.

**Agentic mobility systems (AgMS)** were introduced conceptually by Yu (2025, *Transport Policy*) as the mobility part of agentic transportation systems, in which humans and organisations (*principals*) increasingly delegate decisions to AI agents. This paper gives AgMS a formal mathematical definition and develops its theory. In the formulation, agents interpret goals stated in natural language, reason over memory and tools, communicate, and act through graded authority: self-execution, approval by a human or by another agent, and safety shields. The paper proves that such systems are well posed and admit a Markov lift. It also characterises their long-run behaviour, decomposes the welfare loss of delegation by layer, and gives conditions for shield-certified safety.

> Yu, J. (2025). Preparing for an agentic era of human-machine transportation systems: Opportunities, challenges, and policy recommendations. *Transport Policy*, 171, 78–97. https://doi.org/10.1016/j.tranpol.2025.05.030

This repository contains a small reference implementation of the AgMS protocol with *language-mediated surrogate agents*. These are explicit stochastic kernels that reproduce the structural features of language-model agents, so every theorem can be checked exactly and every run replayed. It reproduces Experiments 1 to 3 of Section 9 of the paper exactly. For Experiment 4, in which a language model takes the place of the surrogate, it contains the script, the results, and the log of every request to the model. Section 8 of the paper defines a catalogue of performance measures for AgMS simulators, covering system performance, safety, delegation, and the justifiability, inclusivity, and adaptability (JIA) principles. A scorecard table at the end of Section 9 restates Experiments 1 to 3 as such measures.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the three experiments (about four minutes on a laptop CPU) and writes the figures (PDF) and numerical results (JSON) to `results/`. The files already in `results/` are the outputs reported in the paper. Rerunning reproduces them exactly, because every random draw is seeded. The paper's figures were rendered with the environment variable `AGMS_USETEX=1`, which requires a LaTeX installation; without it, the scripts use matplotlib's built-in Computer Modern fonts and produce the same numbers.

Each experiment can also be run on its own:

```bash
python code/exp1_ergodicity.py      # Experiment 1: memory, temperature and long-run behaviour
python code/exp2_accountability.py  # Experiment 2: accountability decomposition and effective influence
python code/exp3_safety.py          # Experiment 3: shields, mediation and composition
```

To redraw Figure 6 from the saved results without rerunning the replications, use `python code/exp2_accountability.py --plot-only`.

### Experiment 4: a language model in the loop

Experiments 1 to 3 use surrogate agents and need no account anywhere. `code/exp4_llm_agents.py` replaces the surrogate by calls to a language model: on every day on which an agent re-plans, the model reads the traveller's instruction in plain words, picks an objective profile, and chooses a route. The script then compares the model with the surrogate on the inputs the system visited (drift rate, invalid outputs, route choices, total-variation distance from repeated calls) and compares system outcomes on the same seeds.

The results in the paper (Section 9.4, Table 8) were obtained with OpenAI `gpt-6-luna` through the Responses API with default settings on 11 November 2026 (4,822 requests). Calls to a hosted model are not reproducible, so the record is the log:

- `results/exp4_llm_log_gpt-6-luna.jsonl`: every request and reply
- `results/exp4_llm_results_gpt-6-luna.json`: the results computed from it

To recompute the results from the log, with no account and no new calls:

```bash
python code/exp4_llm_agents.py --analyse-only --tag gpt-6-luna
```

To run the experiment again, or with another model:

```bash
pip install openai
export OPENAI_API_KEY=sk-...                       # PowerShell: $env:OPENAI_API_KEY = "sk-..."
python code/exp4_llm_agents.py --list-models       # model names your key can use
python code/exp4_llm_agents.py --model MODEL_NAME  # about 4,800 short calls with the default settings
```

- The key is read from the environment and is never written to a file.
- Every call is appended to `results/exp4_llm_log_<model>.jsonl`. If the run stops, run the same command again and it resumes. Language-model calls are not reproducible, so this log is the record: `--analyse-only` recomputes the results from it without new calls.
- A short trial first: add `--seeds 1 --days 3 --rep-inputs 5 --rep-calls 3 --tag trial` (about 45 calls).
- `python code/exp4_llm_agents.py --mock` runs the whole pipeline offline with a stand-in for the model.
- Logs and results are named after the model. A run with a name that already has a complete log makes no new calls; add `--tag NAME` for a fresh run.
- `run_all.py` does not run this experiment.

## Repository structure

| Path | Content | Paper |
|---|---|---|
| `code/agms_core.py` | Reference implementation of one AgMS tick: specification, interpretation, deliberation, actuation (self-execution, approval, shield), substrate, memory | Sections 3 and 9, Algorithm 1 |
| `code/exp1_ergodicity.py` | Exact lifted Markov chain of the two-route example (response map, invariant law across temperatures, mixing-time phase diagram); private forgetful vs non-forgetful memories (belief mechanism, lock-in diagram) | Theorem 3, Example 2, Figures 4–5 |
| `code/exp2_accountability.py` | Layer-wise accountability decomposition with counterfactual replay under common random numbers (two orders and the order-free Shapley attribution, two regimes, layer sweeps, individual losses with a static softmax prediction); effective influence; rubber-stamp equivalence | Theorem 5, Proposition 1, Figure 6 |
| `code/exp3_safety.py` | Validity vs permit (product-form) shields; misaligned agents; persuasion of unmediated human drivers | Theorem 6, Table 7 |
| `code/exp4_llm_agents.py` | The same AgMS with a language model as every agent's interpretation and deliberation; fidelity of the surrogate on visited inputs; resumable call log | Corollary 4, Table 8 |
| `results/` | Figures and JSON results reported in the paper, and the call log of Experiment 4 | Section 9 |

## Parameter names

The code uses the paper's notation:

| Code | Paper | Meaning |
|---|---|---|
| `q_I` | $q_I$ | probability that the interpretation kernel drifts to a non-faithful objective |
| `q_D` | $q_D$ | probability that deliberation emits an invalid output |
| `alpha` | $\alpha$ | probability that the reviewer (here, the traveller) approves a proposed switch |
| `T` | $T$ | sampling temperature of deliberation |
| `rho` | $\rho$ | re-planning probability per day |
| `L` | $L$ | memory window (days) |
| `disc` | $\beta$ | discount factor |

The layer-reference flags `perfect_spec`, `faithful`, `no_halluc` and `full_exec` replace the specification, interpretation, deliberation and actuation layers with their reference kernels. They accept a boolean or a per-agent mask, so any layer of any subset of agents can be substituted and replayed. Each stochastic layer draws from its own random stream, so substitutions use common random numbers.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT. See `LICENSE`.
