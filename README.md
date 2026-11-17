# Supplementary code and results

Manuscript: *A Markov Model of Agentic Mobility Systems: From Human–Machine Agency to Network Traffic Dynamics*
Jiangbo Yu and Ali Eslami, McGill University (2026). Preprint.

This package contains a small reference implementation of the agentic mobility system of the
manuscript (Algorithm 1) with surrogate agents, the scripts for the five experiments of Section 8,
the results reported in the paper, and the log of every language-model request of Experiment 4.

## Requirements

Python 3.10 or newer, `pip install -r requirements.txt` (numpy, scipy, matplotlib, networkx).
The figures in the paper were drawn with `AGMS_USETEX=1`, which needs a LaTeX installation; without it
the scripts use matplotlib's built-in fonts and give the same numbers.

## Experiments 1-3 and 5

```bash
python run_all.py                     # about four minutes on a laptop
python code/exp1_ergodicity.py        # Experiment 1: memory, temperature, long-run behavior (Theorem 4, Example 2)
python code/exp2_accountability.py    # Experiment 2: delegation-loss decomposition and herding (Theorem 5, Proposition 1)
python code/exp3_enforcement.py       # Experiment 3: enforcing a bridge load limit (Theorem 6)
python code/exp5_network.py           # Experiment 5: decomposition and herding on the Sioux Falls network (Theorem 5)
```

Every random draw is seeded, so rerunning reproduces `results/` exactly. Each stochastic layer has its own
random stream, so any layer of any set of agents can be replaced by a reference version and replayed with
common random numbers. `python code/exp2_accountability.py --plot-only` redraws Figure 5 from the saved results, and
`python code/exp5_network.py --plot-only` redraws Figure D.2.

The Sioux Falls network data in `data/siouxfalls/` are taken unchanged from the Transportation Networks
for Research repository (https://github.com/bstabler/TransportationNetworks); see that repository for their terms.

## Experiment 4: a language model in the loop

`code/exp4_llm_agents.py` replaces the surrogate interpretation and deliberation by requests to a hosted
language model. The run reported in the paper used OpenAI `gpt-6-luna` through the Responses API with default
settings on 11 November 2026 (4,822 requests). Hosted models are not reproducible, so the log is the record:

- `results/exp4_llm_log_gpt-6-luna.jsonl`: every request and reply
- `results/exp4_llm_results_gpt-6-luna.json`: the results computed from it (the key `corollary_4` refers to Corollary 3 of the manuscript)

Recompute the results from the log, with no account and no new calls:

```bash
python code/exp4_llm_agents.py --analyse-only --tag gpt-6-luna
```

Run it again, or with another model (the key is read from the environment and never written to a file):

```bash
pip install openai
export OPENAI_API_KEY=...
python code/exp4_llm_agents.py --model MODEL_NAME     # about 4,800 short requests
python code/exp4_llm_agents.py --mock                 # offline stand-in for testing
```

## Parameter names

`q_I` interpretation drift probability; `q_D` invalid-output probability; `alpha` approval probability;
`T` sampling temperature; `rho` re-planning probability; `L` memory window (days); `disc` discount factor.
