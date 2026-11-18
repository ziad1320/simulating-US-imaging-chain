# Plane-Wave Ultrasound Imaging Chain Simulation (MUST / pymust)

A complete, end-to-end simulation of an ultrasound imaging system in Python: probe, pulse, plane-wave transmission, scattering, raw channel data, delay-and-sum beamforming, B-mode image, and measurements. An optional extra applies Null Subtraction Imaging (NSI) to the same data.

---

## 1. Overview

**Goal.** Show, step by step, how an ultrasound image is made, and how each step changes the final image. Every stage is software, so every stage can be plotted and measured.

**What is simulated**

| Stage | Choice |
|---|---|
| Probe | Linear array, 128 elements, pitch 0.3048 mm, element width 0.2698 mm, centre frequency 5 MHz, bandwidth about 3-7 MHz (like an L9-4/38) |
| Sampling / sound speed | 80 MHz / 1540 m/s |
| Transmit | Plane wave at 0 degrees, or 21 angles from -20 to +20 degrees (compounding) |
| Targets | Point, wires (with a close pair), and a speckle background with one anechoic cyst and one hyperechoic cyst |
| Receive | 32-element sub-aperture (9.75 mm) on each scan line |
| Beamforming | Own delay-and-sum (DAS) code with any weights |
| Image | Envelope (Hilbert), normalize, log compression, 60 dB display (40 dB for the close pair) |
| Optional | Null Subtraction Imaging (NSI) from Agarwal, Reeg, Podkowa and Oelze, IEEE TUFFC 66(1), 2019 |

**Rule of the project.** Artifacts (speckle, clutter, NSI side effects) are shown and explained. They are not fixed.

**Team** (course final project): Ziad Ramadan Mohamed, Anas Mohamed Abdalaziz, Sherif Mohamed Mahmoud, Mostafa Ashraf Mostafa, Mazen Atef Awad.

**Repository layout**

```
simulating-US-imaging-chain/
|-- data/                   generated RF data (.npz, git-ignored) and metrics (.json)
|-- figures/                generated pictures (.png)
|-- docs/                   project_explanation.tex (LaTeX explanation), Final_Project_Proposal.pdf
|-- src/
|   |-- lib/                the library (an importable package called "lib")
|   |   |-- __init__.py
|   |   |-- config.py       all parameters (change numbers only here)
|   |   |-- phantoms.py     scatterer lists for the 3 phantoms
|   |   |-- beamformer.py   own DAS beamformer, compounding, envelope, NSI
|   |   `-- metrics.py      FWHM, sidelobe, CR, CNR, speckle SNR
|   `-- scripts/            the programs you run
|       |-- 01_probe_pulse.py    Step 1: array and pulse
|       |-- 02_phantoms.py       Step 2: phantom plots
|       |-- 03_simulate.py       Step 3: RF data generation (saves data/*.npz)
|       |-- 04_das.py            Step 4+7: DAS images and measurements
|       |-- 05_nsi.py            Step 5 (optional): NSI
|       |-- 06_validate.py       automatic tests (PASS/FAIL)
|       `-- run_all.py           runs everything in order
|-- LICENSE                 MIT licence
|-- README.md
|-- requirements.txt        Python packages
|-- pyproject.toml          lets "pip install -e ." make the "lib" package importable
`-- .gitignore
```

---

## 2. Simulation Explanation

### The chain, in order

1. **Probe and pulse (MUST).** `config.make_param()` builds the probe. Each element turns an electric pulse into pressure and turns echoes back into voltage. The simulated pulse-echo has a -6 dB band of 3.0-6.47 MHz, with its peak near 4 MHz.
2. **Plane-wave transmission (MUST, `txdelay`).** Element *i* fires with delay `x_i * sin(theta) / c` (shifted so no delay is negative). At 0 degrees, all delays are zero.
3. **Propagation and scattering (MUST, `simus`).** Each phantom is a list of scatterers `(x, z, reflection coefficient)`. `simus` adds the echoes of all scatterers at every element. This uses the linear wave model (the same idea as Field II). The result is the raw RF data, shape `(time, 128 elements)`.
4. **Reception.** The raw data is saved as `rf[angle, time, channel]`. Optional white Gaussian noise (35 dB or 10 dB SNR) is added to a copy.
5. **Beamforming (own code, `beamformer.py`).** For each pixel `(x, z)`:
   - time down: `t_tx = (x sin(theta) + z cos(theta)) / c + t_off`
   - time back to element *i*: `t_rx,i = sqrt((x - x_i)^2 + z^2) / c`
   - read each of the 32 channels at `t_tx + t_rx,i` (linear interpolation), multiply by weight `w_i`, add.

   `t_off = -min(x_i sin(theta)) / c` matches MUST's rule that the first element to fire defines t = 0. If a real scatterer is at the pixel, all 32 values line up and add up. If not, they cancel.
6. **Compounding.** For many angles, the RF images are added **before** the envelope. This keeps the phase, so real echoes add together.
7. **Image formation.** Envelope `|y + j*Hilbert(y)|` along depth, then `20*log10(E / max)`, clipped at -60 dB.
8. **Evaluation (`metrics.py`).**
   - FWHM: width of the beam at -6 dB (lateral and axial)
   - Peak sidelobe: highest level outside the main lobe
   - Contrast ratio: `CR = 20*log10(mean_inside / mean_background)`
   - `CNR = |mean_in - mean_bg| / sqrt(var_in + var_bg)`
   - Speckle SNR: mean / standard deviation of the background (about 1.91 for fully developed speckle)

### The phantoms

| Phantom | Content |
|---|---|
| Point | One scatterer at x = 0, z = 19 mm |
| Wires | Wires at z = 10, 20, 30 mm on the centre line, a close pair at z = 18 mm and x = +/-0.4 mm (0.8 mm apart), plus a weak speckle background |
| Contrast | Random scatterers in x = +/-15 mm, z = 10-40 mm (40 per mm^2 at full density, about 14 per resolution cell). Anechoic cyst: centre (-6, 20) mm, radius 4 mm, strength x0. Hyperechoic cyst: centre (+6, 30) mm, radius 4 mm, strength x4 (+12 dB). |

### Optional: Null Subtraction Imaging (NSI)

Three weight vectors on the 32 elements: ZM (+1 on the first 16, -1 on the last 16), DC1 = ZM + c, DC2 = DC1 mirrored left to right. The same RF data is beamformed three times, and:

`E_NSI = (E_DC1 + E_DC2) / 2 - E_ZM`

Negative values are clipped to a small floor so the log is defined. Smaller `c` gives a narrower beam.

### Limits of the model

- No tissue attenuation, so no depth-gain compensation is used.
- The simulation is 2-D by default (`USE_3D = False` in `config.py`), so the 19 mm elevation focus is ignored. Set `USE_3D = True` to include it (slower).
- Linear acoustics only (no harmonics, no non-linear effects).

---

## 3. How to Setup

**Requirements**

- Python 3.9 or newer (tested with Python 3.12)
- Packages: `pymust`, `numpy`, `scipy`, `matplotlib`
- Disk: about 250 MB for the full 21-angle data (three contrast files of about 61 MB each)
- RAM: 4 GB or more is enough
- CPU: one core is enough. The simulation is the slow part (see the Runbook).

**Install**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # once: makes the "lib" package importable
```

**Check the install** (should print a probe description):

```bash
python -c "import pymust; print(pymust.getparam('L11-5v'))"
```

Run every command from the repository root. The scripts import the `lib` package (`from lib import config`), which is why `pip install -e .` is needed once, in the same environment you run the scripts with. Data and figures are always written to `data/` and `figures/` in the repository root, whatever folder you start from.

---

## 4. Runbook

Run in this order. Each step needs the files made by the steps before it.

```bash
python src/scripts/01_probe_pulse.py
python src/scripts/02_phantoms.py
python src/scripts/03_simulate.py --quick --noise
python src/scripts/04_das.py
python src/scripts/05_nsi.py
python src/scripts/06_validate.py
```

Or run everything: `python src/scripts/run_all.py --quick --noise`

**Choose a simulation size for step 3**

| Command | Contrast phantom | Time (1 core) |
|---|---|---|
| `python src/scripts/03_simulate.py --quick --noise` | 5 angles, half density | about 4 min |
| `python src/scripts/03_simulate.py --medium --noise` | 21 angles, half density | about 15 min |
| `python src/scripts/03_simulate.py --noise` | 21 angles, full density (project spec) | about 40 min |

The point and wire phantoms take seconds in every mode. Add `--only point`, `--only wires` or `--only contrast` to simulate one phantom. The `--noise` flag also saves the 35 dB and 10 dB noisy copies.

**Run steps 4 and 5 for one phantom only**

```bash
python src/scripts/04_das.py point        # or: wires, contrast
python src/scripts/05_nsi.py point
```

**Beamforming time.** The 21-angle contrast image takes a few minutes in `04_das.py` and again in `05_nsi.py`.

**Tips**

- Change any number (angles, density, cyst size, NSI `c`) in `src/lib/config.py`, then rerun from step 2 or 3.
- Re-running `03_simulate.py` overwrites the files in `data/`.
- The `.npz` data files are git-ignored (the 21-angle contrast files are about 61 MB each). After cloning the repository, create them again with step 3. The small `metrics_*.json` files and the figures are kept in the repository.
- Expect `06_validate.py` to print `39/39 checks passed` when all stages were run. The NSI checks are skipped if `05_nsi.py` was not run.
- Common problem: `ModuleNotFoundError: No module named 'lib'`. Run `pip install -e .` once from the repository root, in the same environment (virtual environment) that you use to run the scripts.
- Common problem: `AttributeError` or option errors from `simus`. Use the plain `simus` call as in `src/scripts/03_simulate.py`. Extra options such as `ElementSplitting` and `dBThresh` crashed in the tested pymust version.

---

## 5. Output

### Data (`data/`)

| File | Content |
|---|---|
| `point.npz`, `wires.npz`, `contrast.npz` | `rf[angle, time, channel]` (float32), `angles`, `fs`, `fc`, `c`, `pitch`, `n_el`, scatterer positions and strengths. Time starts at 0, sampled at 80 MHz. |
| `wires_snr35.npz`, `wires_snr10.npz`, `contrast_snr35.npz`, `contrast_snr10.npz` | Same data with noise (only with `--noise`) |
| `metrics_04.json` | All DAS measurements |
| `metrics_05.json` | All NSI measurements |

### Figures (`figures/`) and expected results

| Figure | Shows | Expected result |
|---|---|---|
| `fig01_array.png` | 128 elements, 32-element window | Window = 9.75 mm |
| `fig02_pulse_spectrum.png` | Eq. 5 pulse and simulated pulse, time and spectrum | Eq. 5 band 3.00-7.00 MHz. Simulated band 3.0-6.47 MHz. |
| `fig03_phantoms.png` | Scatterer maps, three panels. Point: zoomed 4 x 4 mm window with the single scatterer in red. Wires: the 5 wires (rc = 1) in large red dots over the weak background in small grey dots. Contrast: dot size follows scatterer strength, and zero-strength scatterers are not drawn, so the anechoic cyst (blue circle) looks empty while the hyperechoic cyst (red circle) looks dense and dark. | Point 1, wires 4485 (5 wires + 4480 background), contrast about 36,000 (full density). The contrast title counts every scatterer, including the zero-strength ones inside the anechoic cyst. |
| `fig04_channel_data.png` | Raw RF, time vs element | One curved line per wire |
| `fig05_point_psf.png` | Point image, lateral and axial profiles, own DAS vs `pymust.dasmtx`, theory curve | Point at 19.00 mm. Lateral FWHM 0.79 mm. Axial FWHM 0.24 mm. Peak sidelobe -19.5 dB. Own DAS matches MUST to 0.16 dB RMS. |
| `fig06_wires.png` | Wire image (60 dB), close pair (40 dB), pair profile | Wires at 10.00, 19.99, 30.00 mm. The 0.8 mm pair is not separated (valley -0.3 dB). |
| `fig07_contrast.png` | Cysts, 1 angle vs 21 angles | Hyperechoic CR +11.4 dB (1 angle), +12.0 dB (21 angles). Anechoic CR -15.2 dB, then -23.6 dB. |
| `fig08_nsi_point.png` | NSI point profiles for c = 0.01, 0.03, 0.05 | FWHM 21 um at c = 0.03 (DAS: 793 um). Width grows with c. Peak about 31 dB below DAS. |
| `fig09_nsi_pair.png` | Close pair, DAS vs NSI | NSI makes one peak between the wires (an artifact, documented not fixed) |
| `fig10_nsi_contrast.png` | Cysts, DAS vs NSI | CNR of the hyperechoic cyst drops from 1.15 to 0.34. Speckle SNR drops from 1.51 to 0.44. |

The values above were measured with the 21-angle, half-density run (`--medium`). At full density the contrast and speckle numbers will differ somewhat. Speckle SNR is below 1.91 at half density because there are fewer scatterers per resolution cell.

### Test report (`06_validate.py`)

It prints one line per check, such as `[PASS] lateral FWHM (-6 dB) [mm]  0.793 mm (expected 0.65 .. 0.95)`. The checks cover:

- geometry
- the transmit-time offset at -20, -10, 0, +10, +20 degrees
- point position and size
- the comparison with MUST
- wire depths
- noisy-data localization
- contrast ratios and speckle SNR
- NSI behavior (narrower beam, lower sidelobes, lower CNR and speckle SNR)

---

## 6. Licence

The code in this repository is released under the **MIT licence**. The full text is in the [`LICENSE`](LICENSE) file.

---

## 7. Acknowledgments

- **MUST / pymust** by Damien Garcia (Creatis, Lyon): the open-source ultrasound simulator used for transmit delays and RF simulation.
- **Reference paper:** A. Agarwal, J. Reeg, A. S. Podkowa, and M. L. Oelze, "Improving spatial resolution using incoherent subtraction of receive beams having different apodizations," *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, vol. 66, no. 1, pp. 5-17, Jan. 2019, doi:10.1109/TUFFC.2018.2876285.
- **Python tools:** NumPy, SciPy and Matplotlib.
- The course instructors and teaching staff for the project topic and guidance.
