"""Run the whole chain. python run_all.py [--quick] [--noise]  (use --quick unless you have ~40 min)"""
import subprocess, sys
flags = [a for a in sys.argv[1:] if a in ("--quick", "--noise")]
for cmd in (["01_probe_pulse.py"], ["02_phantoms.py"], ["03_simulate.py"] + flags, ["04_das.py"], ["05_nsi.py"], ["06_validate.py"]):
    print(f"\n######## {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True)
