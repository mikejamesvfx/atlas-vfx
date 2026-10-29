"""Build Physical Sky's Hosek-Wilkie fallback data from the authors' BSD-3 reference source, reproducibly.

Source: the Hosek-Wilkie sample implementation v1.4a (Copyright (c) 2012-2013 Lukas Hosek and Alexander Wilkie,
BSD-3-Clause) as pbrt-v4 carries it, at a pinned commit. Each file's sha256 must match the pins below. The C arrays
are read with a numbers-only parser (no eval, no compile) into:

  atlas_vfx/layers/data/hosek_wilkie_spectral.npz    float64 arrays (np.savez_compressed, no pickle)
  atlas_vfx/layers/data/hosek_wilkie_spectral.json   manifest: per-array sha256 of the raw bytes, shapes, source
  atlas_vfx/layers/data/THIRD_PARTY_HOSEK_WILKIE.md  the licence, verbatim from the source header

Arrays (11 bands, 320-720 nm every 40 nm):
  sky_config   (11, 2 albedo, 10 turbidity, 6 Bezier points, 9 params)   datasetNNN
  sky_radiance (11, 2 albedo, 10 turbidity, 6 Bezier points)              datasetRadNNN
  solar        (11, 10 turbidity, 45 pieces, 4 coefficients, file order)  solarDatasetNNN
  limb         (11, 6)                                                    limbDarkeningDatasetNNN

--golden also compiles the reference C (MSVC via vswhere on Windows, else cc/gcc/clang) with a small harness and
writes tests/fixtures/hosek_golden.json: reference sky and solar radiance values that tests/test_sky_hosek.py holds
the numpy port to. Owner's tool; the shipped package never runs it.

Run: <python> tools/hosek_wilkie_data.py [--from FOLDER] [--golden]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "atlas_vfx" / "layers" / "data"
GOLDEN = ROOT / "tests" / "fixtures" / "hosek_golden.json"
COMMIT = "9772673e43e74b931675081b2826f4acc91ba348"
BASE = f"https://raw.githubusercontent.com/mmp/pbrt-v4/{COMMIT}/src/ext/skymodel/"
PINS = {
    "ArHosekSkyModel.c": "16cca27268ed1cf27067edef8e9526e6ffa157b0e0467dab422bfdd13ff21fe8",
    "ArHosekSkyModel.h": "0f51b75b7207f336224f0154c58be99537e419c89d26a51add34aa1db6209774",
    "ArHosekSkyModelData_Spectral.h": "46459aefe7cfa14759495200b107a0db7ac0f8d5cec8792582123286ddf0cc09",
    "ArHosekSkyModelData_CIEXYZ.h": "cd879dc216340d9416b16930cb92f4bd3e3bbcb7c6f0a3ce692bf006f4110320",
    "ArHosekSkyModelData_RGB.h": None,   # only needed to compile the reference (--golden); hash recorded, not pinned
}
BANDS = list(range(320, 721, 40))
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
ARRAY = re.compile(r"double\s+(\w+)\s*\[\s*\]\s*=\s*\{([^{}]*)\}\s*;")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(src: Path | None, need: list[str]) -> tuple[Path, dict]:
    """The source files in a fresh folder, each checked against its pin."""
    work = Path(tempfile.mkdtemp(prefix="hosek_src_"))
    got = {}
    for name in need:
        if src:
            b = (src / name).read_bytes()
        else:
            with urllib.request.urlopen(BASE + name, timeout=60) as r:
                b = r.read()
        h = sha256(b)
        if PINS.get(name) and h != PINS[name]:
            raise SystemExit(f"{name}: sha256 {h} does not match the pin {PINS[name]}")
        (work / name).write_bytes(b)
        got[name] = {"url": BASE + name, "sha256": h, "bytes": len(b)}
    return work, got


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def parse_arrays(text: str) -> dict[str, np.ndarray]:
    out = {}
    for m in ARRAY.finditer(strip_comments(text)):
        body = m.group(2)
        nums = NUM.findall(body)
        rest = NUM.sub(" ", body).replace(",", " ").split()
        if rest:
            raise SystemExit(f"{m.group(1)}: unexpected tokens {rest[:5]}")
        out[m.group(1)] = np.array([float(n) for n in nums], np.float64)
    return out


def build(arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def take(prefix, size):
        rows = []
        for b in BANDS:
            a = arrs[f"{prefix}{b}"]
            if a.size != size:
                raise SystemExit(f"{prefix}{b}: {a.size} values, expected {size}")
            rows.append(a)
        return np.stack(rows)
    return {
        "bands_nm": np.array(BANDS, np.float64),
        "sky_config": take("dataset", 2 * 10 * 6 * 9).reshape(11, 2, 10, 6, 9),
        "sky_radiance": take("datasetRad", 2 * 10 * 6).reshape(11, 2, 10, 6),
        "solar": take("solarDataset", 10 * 45 * 4).reshape(11, 10, 45, 4),
        "limb": take("limbDarkeningDataset", 6),
    }


def licence_text(c_source: str) -> str:
    m = re.match(r"\s*/\*(.*?)\*/", c_source, flags=re.S)
    if not m or "3-clause BSD" not in m.group(1):
        raise SystemExit("ArHosekSkyModel.c: licence header not found")
    return m.group(1).strip("\n")


def write_data(arrays: dict, files: dict, licence: str) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    (DATA / "hosek_wilkie_spectral.npz").write_bytes(buf.getvalue())
    manifest = {
        "model": "Hosek-Wilkie skylight 2012 + solar radiance 2013, v1.4a, spectral",
        "licence": "BSD-3-Clause, Copyright (c) 2012-2013 Lukas Hosek and Alexander Wilkie",
        "source": {"repo": "github.com/mmp/pbrt-v4", "commit": COMMIT, "path": "src/ext/skymodel", "files": files},
        "built": date.today().isoformat(), "tool": "tools/hosek_wilkie_data.py",
        "units": "spectral radiance W / (m^2 sr nm) at bands_nm",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": sha256(np.ascontiguousarray(v).tobytes())}
                   for k, v in arrays.items()},
    }
    (DATA / "hosek_wilkie_spectral.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                                     newline="\n")
    notice = (
        "# Hosek-Wilkie sky model data\n\n"
        "`hosek_wilkie_spectral.npz` holds the coefficients of the Hosek-Wilkie analytic skylight model (2012) and\n"
        "its solar radiance function (2013), version 1.4a, converted unchanged from the authors' sample\n"
        f"implementation as carried by pbrt-v4 at commit {COMMIT} (`src/ext/skymodel`). The file hashes are in\n"
        "`hosek_wilkie_spectral.json`. `atlas_vfx/layers/sky_hosek.py` is a numpy port of the same formulas.\n\n"
        "L. Hosek, A. Wilkie, \"An Analytic Model for Full Spectral Sky-Dome Radiance\", SIGGRAPH 2012.\n"
        "L. Hosek, A. Wilkie, \"Adding a Solar Radiance Function to the Hosek Skylight Model\", IEEE CG&A 2013.\n\n"
        "The source's licence, verbatim:\n\n```\n" + licence + "\n```\n")
    (DATA / "THIRD_PARTY_HOSEK_WILKIE.md").write_text(notice, encoding="utf-8", newline="\n")
    return manifest


# ------------------------------------------------------------------------------------------------ golden values

HARNESS = r"""
#include <stdio.h>
#include "ArHosekSkyModel.c"
int main(void) {
    double t, a, se, th, ga, wl, el;
    char kind[8];
    while (scanf("%7s", kind) == 1) {
        if (kind[0] == 's') {          /* sky: T albedo sun_el theta gamma wl */
            if (scanf("%lf %lf %lf %lf %lf %lf", &t, &a, &se, &th, &ga, &wl) != 6) return 2;
            ArHosekSkyModelState *s = arhosekskymodelstate_alloc_init(se, t, a);
            printf("%.17g\n", arhosekskymodel_radiance(s, th, ga, wl));
            arhosekskymodelstate_free(s);
        } else {                       /* sun: T albedo ray_el gamma wl */
            if (scanf("%lf %lf %lf %lf %lf", &t, &a, &el, &ga, &wl) != 5) return 2;
            ArHosekSkyModelState *s = arhosekskymodelstate_alloc_init(0.3, t, a);
            printf("%.17g\n", arhosekskymodel_solar_radiance_internal2(s, wl, el, ga));
            arhosekskymodelstate_free(s);
        }
    }
    return 0;
}
"""


def golden_cases():
    rng = np.random.default_rng(20261029)
    sky = []
    for t in (1.0, 1.7, 2.5, 4.3, 7.9, 10.0):
        for a in (0.0, 0.25, 1.0):
            for _ in range(11):
                se = float(np.radians(rng.choice([0.5, 3.0, 12.0, 30.0, 60.0, 89.0])))
                th = float(rng.uniform(0.0, np.pi / 2 - 1e-3))
                ga = float(rng.uniform(0.0, np.pi))
                wl = float(rng.choice([320.0, 345.5, 400.0, 555.0, 600.0, 699.9, 720.0]))
                sky.append([t, a, se, th, ga, wl])
    sun = []
    for t in (1.0, 2.2, 5.5, 9.4, 10.0):
        for el_deg in (0.3, 2.0, 7.5, 25.0, 50.0, 88.0):
            for g_deg in (0.0, 0.1, 0.2, 0.25):
                wl = float(rng.choice([320.0, 380.0, 555.0, 640.0, 720.0]))
                sun.append([t, 0.25, float(np.radians(el_deg)), float(np.radians(g_deg)), wl])
    return sky, sun


def msvc_env() -> dict | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    inst = subprocess.run([str(vswhere), "-latest", "-property", "installationPath"], capture_output=True,
                          text=True).stdout.strip()
    vcvars = Path(inst) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        return None
    # cmd is the only way to load vcvars; the path comes from vswhere, nothing user-typed reaches this line
    out = subprocess.run(["cmd", "/d", "/c", "call", str(vcvars), ">nul", "&&", "set"], capture_output=True,
                         text=True).stdout
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return env if "INCLUDE" in env else None


def compile_reference(work: Path) -> Path:
    (work / "golden.c").write_text(HARNESS, encoding="utf-8")
    exe = work / ("golden.exe" if os.name == "nt" else "golden")
    env = msvc_env() if os.name == "nt" else None
    if env:
        cl = shutil.which("cl", path=env.get("PATH") or env.get("Path"))
        subprocess.run([cl, "/nologo", "/O2", "/fp:precise", "/w", "golden.c", f"/Fe{exe.name}"], cwd=work, env=env,
                       check=True, capture_output=True)
    else:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            raise SystemExit("--golden needs a C compiler (MSVC, cc, gcc or clang)")
        subprocess.run([cc, "-O2", "-w", "golden.c", "-o", str(exe), "-lm"], cwd=work, check=True)
    return exe


def write_golden(work: Path, files: dict) -> None:
    exe = compile_reference(work)
    sky, sun = golden_cases()
    lines = [("s " + " ".join(repr(v) for v in c)) for c in sky] + [("u " + " ".join(repr(v) for v in c)) for c in sun]
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    vals = [float(v) for v in out.stdout.split()]
    if len(vals) != len(lines):
        raise SystemExit(f"reference gave {len(vals)} values for {len(lines)} cases")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({
        "about": "Hosek-Wilkie v1.4a reference C (compiled from the pinned source) evaluated at fixed cases; "
                 "angles in radians, wavelength nm, radiance W/(m^2 sr nm). sky: [T, albedo, sun_el, theta, gamma, "
                 "wl, value] from arhosekskymodel_radiance; sun: [T, albedo, ray_el, gamma, wl, value] from "
                 "arhosekskymodel_solar_radiance_internal2.",
        "source_commit": COMMIT, "source_sha256": {k: v["sha256"] for k, v in files.items()},
        "built": date.today().isoformat(),
        "sky": [c + [v] for c, v in zip(sky, vals[:len(sky)])],
        "sun": [c + [v] for c, v in zip(sun, vals[len(sky):])],
    }, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN.relative_to(ROOT)}: {len(sky)} sky + {len(sun)} sun values")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", type=Path, help="local folder with the source files instead of downloading")
    ap.add_argument("--golden", action="store_true", help="also compile the reference C and write the golden values")
    a = ap.parse_args(argv)
    need = ["ArHosekSkyModel.c", "ArHosekSkyModel.h", "ArHosekSkyModelData_Spectral.h"]
    if a.golden:
        need += ["ArHosekSkyModelData_CIEXYZ.h", "ArHosekSkyModelData_RGB.h"]
    work, files = fetch(a.src, need)
    arrays = build(parse_arrays((work / "ArHosekSkyModelData_Spectral.h").read_text(encoding="utf-8")))
    m = write_data(arrays, files, licence_text((work / "ArHosekSkyModel.c").read_text(encoding="utf-8")))
    print(f"wrote {DATA.relative_to(ROOT)}: " + ", ".join(f"{k} {v['shape']}" for k, v in m["arrays"].items()))
    if a.golden:
        write_golden(work, files)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
