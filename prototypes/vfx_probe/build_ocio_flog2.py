"""ACES 2.0 studio config (OCIO built-in studio-config-v4.0.0_aces-v2.0_ocio-v2.5) plus a FUJIFILM F-Log2 F-Gamut
input colour space, which the built-in configs do not ship. Writes the config and checks the curve against
colour-science's reference implementation (Fujifilm F-Log2 data sheet constants).

F-Log2: in >= cut1: out = c*log10(a*in + b) + d ; else out = e*in + f  (in = scene-linear reflection)
F-Gamut: ITU-R BT.2020 primaries, D65 white.
"""
import sys
import numpy as np
import PyOpenColorIO as ocio
import colour
from colour.models.rgb.transfer_functions.fujifilm_f_log import CONSTANTS_FLOG2 as K, log_decoding_FLog2

BASE = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
OUT = sys.argv[1] if len(sys.argv) > 1 else "atlas_aces2.0_studio_flog2.ocio"

cfg = ocio.Config.CreateFromBuiltinConfig(BASE)
ref = cfg.getColorSpace(cfg.getCanonicalName("ACES2065-1"))
assert ref is not None, "ACES2065-1 missing"

# F-Gamut (Rec.2020, D65) -> ACES2065-1 (AP0, ACES white), CAT02 like the other camera spaces in the config
M = colour.matrix_RGB_to_RGB(colour.RGB_COLOURSPACES["F-Gamut"], colour.RGB_COLOURSPACES["ACES2065-1"],
                             chromatic_adaptation_transform="CAT02")
m44 = [*M[0], 0, *M[1], 0, *M[2], 0, 0, 0, 0, 1]

log = ocio.LogCameraTransform(linSideBreak=[K.cut1] * 3, base=10.0,
                              logSideSlope=[K.c] * 3, logSideOffset=[K.d] * 3,
                              linSideSlope=[K.a] * 3, linSideOffset=[K.b] * 3,
                              linearSlope=[K.e] * 3, direction=ocio.TRANSFORM_DIR_INVERSE)
cs = ocio.ColorSpace(referenceSpace=ocio.REFERENCE_SPACE_SCENE, name="F-Log2 F-Gamut",
                     family="Input/FUJIFILM", encoding="log",
                     description="FUJIFILM F-Log2 curve, F-Gamut (BT.2020 primaries, D65). Added by Atlas "
                                 "(not in the OCIO built-in configs). Constants: Fujifilm F-Log2 data sheet, "
                                 "verified against colour-science log_decoding_FLog2.")
cs.addAlias("flog2_fgamut")
cs.setTransform(ocio.GroupTransform([log, ocio.MatrixTransform(matrix=m44)]), ocio.COLORSPACE_DIR_TO_REFERENCE)
cfg.addColorSpace(cs)
cfg.validate()
with open(OUT, "w") as f:
    f.write(cfg.serialize())

# check: OCIO curve vs colour-science on code values 0..1 (grey axis), in ACES2065-1 the grey axis maps to itself
cv = np.linspace(0.05, 1.0, 400, dtype=np.float32)
proc = cfg.getProcessor("F-Log2 F-Gamut", "ACES2065-1").getDefaultCPUProcessor()
rgb = np.repeat(cv[:, None], 3, 1).copy()
proc.applyRGB(rgb)
ref_lin = log_decoding_FLog2(cv, in_normalised_code_value=True, out_reflection=True)
err = np.max(np.abs(rgb[:, 1] - ref_lin) / np.maximum(np.abs(ref_lin), 1e-3))
grey = cfg.getProcessor("F-Log2 F-Gamut", "ACES2065-1").getDefaultCPUProcessor().applyRGB([0.391007, 0.391007, 0.391007])
print(f"wrote {OUT}  (base {BASE})")
print(f"max relative error vs colour-science: {err:.2e}")
print(f"F-Log2 code 0.391 (18% grey) -> ACES2065-1 {np.round(grey, 4).tolist()}")
print("displays/views:", cfg.getDefaultDisplay(), list(cfg.getViews(cfg.getDefaultDisplay())))
