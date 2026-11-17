"""config.py - every parameter of the project lives here (no hard-coded numbers elsewhere)."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------------- transducer (L9-4/38-like linear array) ----------------
C = 1540.0              # speed of sound [m/s]
FC = 5e6                # centre frequency [Hz]
FS = 80e6               # RF sampling frequency [Hz]
N_EL = 128              # number of elements
PITCH = 0.3048e-3       # [m]
WIDTH = 0.2698e-3       # element width [m]
BANDWIDTH_PCT = 80      # pulse-echo -6 dB fractional bandwidth [%]  (3-7 MHz at 5 MHz)
ELEV_FOCUS = 19e-3      # elevation focus [m]  (only used if USE_3D = True)
ELEV_HEIGHT = 6e-3      # element height [m]   (only used if USE_3D = True)
USE_3D = False          # False: 2-D simulation (y = 0, fast). True: include elevation focusing.
LAMBDA = C / FC         # wavelength [m] (~0.308 mm)

# ---------------- transmit / receive ----------------
ANGLES_SINGLE = np.array([0.0])                              # [rad]
ANGLES_COMPOUND = np.deg2rad(np.linspace(-20, 20, 21))       # 21 angles, -20..+20 deg
N_SUB = 32                                                   # receive sub-aperture (elements)
APERTURE = N_SUB * PITCH                                     # ~9.75 mm

# ---------------- image grid ----------------
LATERAL_UPSAMPLE = 8    # scan lines per element pitch (point / wire tests)
LATERAL_UPSAMPLE_CONTRAST = 4
DZ = LAMBDA / 16        # axial pixel size [m] (~19 um -> 40 MHz equivalent sampling)
DR_DB = 60              # display dynamic range [dB]
DR_DB_WIRES_CLOSE = 40  # for the close-wire phantom

# ---------------- phantoms ----------------
POINT_POS = (0.0, 19e-3)                          # (x, z)
WIRE_Z = (10e-3, 20e-3, 30e-3)                    # wires on the centre line
CLOSE_PAIR_Z = 18e-3
CLOSE_PAIR_DX = 0.4e-3                            # pair at x = +/- DX  -> 0.8 mm apart
WIRE_BG_DENSITY_PER_MM2 = 8                       # weak speckle background for the wire phantom
WIRE_BG_RC = 0.01                                 # background amplitude relative to wire (RC = 1)

CONTRAST_X = (-15e-3, 15e-3)
CONTRAST_Z = (10e-3, 40e-3)
SCAT_PER_MM2 = 40                                 # >= 10 scatterers per resolution cell (2-D)
ANECHOIC = (-6e-3, 20e-3, 4e-3)                   # (x, z, radius)
HYPER = (6e-3, 30e-3, 4e-3)
ANECHOIC_FACTOR = 0.0                             # RC multiplier inside the anechoic cyst
HYPER_FACTOR = 4.0                                # RC multiplier inside the hyperechoic cyst (+12 dB)
SEED = 1234

# ---------------- noise ----------------
NOISE_SNR_DB = (35, 10)

# ---------------- NSI ----------------
NSI_C = 0.03
NSI_C_SWEEP = (0.01, 0.03, 0.05)


def make_param():
    """Build the pymust probe structure from the constants above."""
    import pymust
    p = pymust.getparam("L11-5v")
    p.Nelements = N_EL
    p.pitch = PITCH
    p.width = WIDTH
    p.kerf = PITCH - WIDTH
    p.fc = FC
    p.fs = FS
    p.bandwidth = BANDWIDTH_PCT
    p.focus = ELEV_FOCUS
    p.height = ELEV_HEIGHT
    p.c = C
    return p


def element_x():
    """Lateral element positions [m], array centred at x = 0."""
    return (np.arange(N_EL) - (N_EL - 1) / 2) * PITCH
