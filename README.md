# AgMS-foundation

Code and results for the paper

> **Mathematical Foundations of Agentic Mobility Systems: Modelling Human and Machine Agency for Trustworthy Delegation, Autonomy, and Accountability**
> Ali Eslami and Jiangbo Yu, McGill University (2026). Submitted to *Transportation Research Part B: Methodological*.

**Agentic mobility systems (AgMS)** were introduced conceptually by Yu (2025, *Transport Policy*) as the mobility part of agentic transportation systems, in which humans and organisations (*principals*) increasingly delegate decisions to AI agents. This paper gives AgMS a formal mathematical definition and develops its theory. In the formulation, agents interpret goals stated in natural language, reason over memory and tools, communicate, and act through graded authority: self-execution, approval by a human or by another agent, and safety shields. The paper proves that such systems are well posed and admit a Markov lift. It also characterises their long-run behaviour, decomposes the welfare loss of delegation by layer, and gives conditions for shield-certified safety.

> Yu, J. (2025). Preparing for an agentic era of human-machine transportation systems: Opportunities, challenges, and policy recommendations. *Transport Policy*, 171, 78–97. https://doi.org/10.1016/j.tranpol.2025.05.030

This repository contains a small reference implementation of the AgMS protocol with *language-mediated surrogate agents*. These are explicit stochastic kernels that reproduce the structural features of language-model agents, so every theorem can be checked exactly and every run replayed. It reproduces all experiments in Section 9 of the paper. Section 8 of the paper defines a catalogue of performance measures for AgMS simulators, covering system performance, safety, delegation, and the justifiability, inclusivity, and adaptability (JIA) principles. A scorecard table at the end of Section 9 restates the experiments as such measures.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the three experiments (about two minutes on a laptop CPU) and writes the figures (PDF) and numerical results (JSON) to `results/`. The files already in `results/` are the outputs reported in the paper. Rerunning reproduces them exactly, because every random draw is seeded. The paper's figures were rendered with the environment variable `AGMS_USETEX=1`, which requires a LaTeX installation; without it, the scripts use matplotlib's built-in Computer Modern fonts and produce the same numbers.

Each experiment can also be run on its own:

```bash
python code/exp1_ergodicity.py      # Experiment 1: memory, temperature and long-run behaviour
python code/exp2_accountability.py  # Experiment 2: accountability decomposition and effective influence
python code/exp3_safety.py          # Experiment 3: shields, mediation and composition
```

To redraw Figure 6 from the saved results without rerunning the replications, use `python code/exp2_accountability.py --plot-only`.

## Repository structure

| Path | Content | Paper |
|---|---|---|
| `code/agms_core.py` | Reference implementation of one AgMS tick: specification, interpretation, deliberation, actuation (self-execution, approval, shield), substrate, memory | Sections 3 and 9, Algorithm 1 |
| `code/exp1_ergodicity.py` | Exact lifted Markov chain of the two-route example (response map, invariant law across temperatures, mixing-time phase diagram); private forgetful vs non-forgetful memories (belief mechanism, lock-in diagram) | Theorem 3, Example 2, Figures 4–5 |
| `code/exp2_accountability.py` | Layer-wise accountability decomposition with counterfactual replay under common random numbers (two orders, two regimes, layer sweeps, individual losses with a static softmax prediction); effective influence; rubber-stamp equivalence | Theorem 5, Proposition 1, Figure 6 |
| `code/exp3_safety.py` | Validity vs permit (product-form) shields; misaligned agents; persuasion of unmediated human drivers | Theorem 6, Table 7 |
| `results/` | Figures and JSON results reported in the paper | Section 9 |

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
