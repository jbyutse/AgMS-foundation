"""Reproduce Experiments 1-3 and 5 of the paper (Section 8). Experiment 4 uses a hosted language model; see README. Outputs go to results/."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp1_ergodicity.py", "exp2_accountability.py", "exp3_enforcement.py", "exp5_network.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        t0 = time.time()
        print(f"Running {script} ...", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "code", script)], check=True,
                       stdout=subprocess.DEVNULL)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    print("All results written to", os.path.join(HERE, "results"))
