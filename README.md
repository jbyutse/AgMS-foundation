# agms-foundations

Code and results for the paper

> **Agentic Mobility Systems: A Mathematical Foundation for Human and Machine Agency, Delegation, Autonomy, and Accountability**
> Ali Eslami and Jiangbo Yu, McGill University (2026). Submitted to journal

The paper defines **agentic mobility systems (AgMS)**: transportation systems in which humans and organisations (*principals*) delegate decisions to AI agents. The agents interpret goals stated in natural language, reason over memory and tools, communicate, and act through graded authority (self-execution, human approval, safety shields). The paper proves that such systems are well posed and admit a Markov lift. It characterises their long-run behaviour, decomposes the welfare loss of delegation by layer, and gives conditions for shield-certified safety.

This repository contains a small reference implementation of the AgMS protocol with *language-mediated surrogate agents*. These are explicit stochastic kernels that reproduce the structural features of language-model agents, so every theorem can be checked exactly and every run replayed. It reproduces all experiments in Section 8 of the paper.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the three experiments (about two minutes on a laptop CPU) and writes the figures (PDF) and numerical results (JSON) to `results/`. The files already in `results/` are the outputs reported in the paper. Rerunning reproduces them exactly, because every random draw is seeded.

Each experiment can also be run on its own:

```bash
python code/exp1_ergodicity.py      # Experiment 1: memory, temperature and long-run behaviour
python code/exp2_accountability.py  # Experiment 2: accountability decomposition and effective influence
python code/exp3_safety.py          # Experiment 3: shields, mediation and composition
```

## Repository structure

| Path | Content | Paper |
|---|---|---|
| `code/agms_core.py` | Reference implementation of one AgMS tick: specification, interpretation, deliberation, actuation (self-execution, approval, shield), substrate, memory | Sections 3 and 8, Algorithm 1 |
| `code/exp1_ergodicity.py` | Exact lifted Markov chain of the two-route example; private forgetful vs non-forgetful memories | Theorem 3, Example 2, Figures 1--2 |
| `code/exp2_accountability.py` | Layer-wise accountability decomposition with counterfactual replay under common random numbers; effective influence; rubber-stamp equivalence | Theorem 5, Proposition 1, Figure 3 |
| `code/exp3_safety.py` | Validity vs permit (product-form) shields; misaligned agents; persuasion of unmediated human drivers | Theorem 6, Table 4 |
| `results/` | Figures and JSON results reported in the paper | Section 8 |

## Parameter names

The code uses the paper's notation:

| Code | Paper | Meaning |
|---|---|---|
| `q_I` | $q_I$ | probability that the interpretation kernel drifts to a non-faithful objective |
| `q_D` | $q_D$ | probability that deliberation emits an invalid output |
| `alpha` | $\alpha$ | probability that a human reviewer approves a proposed switch |
| `T` | $T$ | sampling temperature of deliberation |
| `rho` | $\rho$ | re-planning probability per day |
| `L` | $L$ | memory window (days) |
| `disc` | $\beta$ | discount factor |

The layer-reference flags `perfect_spec`, `faithful`, `no_halluc` and `full_exec` replace the specification, interpretation, deliberation and actuation layers with their reference kernels. They accept a boolean or a per-agent mask, so any layer of any subset of agents can be substituted and replayed. Each stochastic layer draws from its own random stream, so substitutions use common random numbers.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT. See `LICENSE`.
