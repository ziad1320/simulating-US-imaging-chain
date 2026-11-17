# Ultrasound imaging chain simulation (MUST / pymust)

Install: pip install pymust numpy scipy matplotlib
Run in order (from this folder):
  python 01_probe_pulse.py
  python 02_phantoms.py
  python 03_simulate.py --noise            # full spec, ~40 min   (--medium ~15 min, --quick ~4 min)
  python 04_das.py
  python 05_nsi.py
  python 06_validate.py                    # expect 39/39 checks passed
Figures -> figures/, RF data and metrics -> data/.
