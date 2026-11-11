"""Reproduce all experiments of the paper (Section 9). Outputs are written to results/."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp1_ergodicity.py", "exp2_accountability.py", "exp3_safety.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        t0 = time.time()
        print(f"Running {script} ...", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "code", script)], check=True,
                       stdout=subprocess.DEVNULL)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    print("All results written to", os.path.join(HERE, "results"))
