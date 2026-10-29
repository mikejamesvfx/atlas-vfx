"""T2: Plate Read and Tracking Proxy.

- a synthetic 10-bit 4:2:2 FULL-range file (HEVC Rext lossless, the same format as the X-H2; ProRes cannot carry a
  full-range flag) decodes to the code values written, frame selection is exact
- a LEGAL-range file (ProRes 422 HQ, like the DSCF3000 conversion) holding the same values is labelled legal and
  decodes to its own expansion, not the full-range values; forcing the wrong range shows the prototype's bug
- F-Log2 code 0.391 -> 0.18 ACEScg through the shipped OCIO config (1e-3), straight and through a decoded file
- tracking proxies: one PNG per selected frame, named by plate index, sidecar written; refuses the plate folder
- F-Log (original) code 0.4593 -> 0.18 ACEScg; the Fuji VideoRecordingMode tag (0x3803) picks F-Log / F-Log2
- real plate, if present: DSCF3000 probes as full range, X-H2, F-Log F-Gamut (from its tag), 528 frames,
  TC 03:02:20:07

Run: <V141 python> tests\\test_plate.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ATLAS_VFX_OUT_ROOTS", tempfile.gettempdir())     # proxies may be written under temp

from atlas_vfx.camera_side import plate as pl  # noqa: E402

REAL = Path(r"C:\Users\miike\Videos\atlas-vfx_footage\DSCF3000.MOV")
W, H = 256, 128
GREY_CODE = 400                       # round(0.391007 * 1023): F-Log2 18 % grey, full-range 10-bit
LEFT_CODES = [100, 300, 700, 900]     # one per frame, so frame selection is checked
BODY_TAG = "FUJIFILM DIGITAL CAMERA X-H2"


def make_plate(path: Path, codec: str, rng: str, planes=None) -> Path:
    """yuv422p10le frames. Default: 4 frames, neutral chroma, left half LEFT_CODES[k], right half GREY_CODE.
    planes: list of (Y HxW, Cb Hx(W/2), Cr Hx(W/2)) uint16 code arrays instead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if planes is None:
        planes = []
        for code in LEFT_CODES:
            y = np.full((H, W), GREY_CODE, np.uint16)
            y[:, : W // 2] = code
            c = np.full((H, W // 2), 512, np.uint16)
            planes.append((y, c, c))
    frames = [y.astype(np.uint16).tobytes() + cb.astype(np.uint16).tobytes() + cr.astype(np.uint16).tobytes()
              for y, cb, cr in planes]
    # No -color_range on input or output: with it, ffmpeg inserts a swscale range conversion before the encoder
    # (measured: 900 -> 903, chroma 512 -> 511). The codes go in untouched and the range is set as metadata only.
    if codec == "hevc":
        full = "1" if rng == "pc" else "0"
        enc = ["-c:v", "libx265", "-x265-params", "lossless=1:log-level=error", "-tag:v", "hvc1",
               "-bsf:v", f"hevc_metadata=video_full_range_flag={full}:matrix_coefficients=1"]
    else:
        assert rng == "tv", "ProRes is video range by spec"
        enc = ["-c:v", "prores_ks", "-profile:v", "3"]
    ph, pw = planes[0][0].shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-s", f"{pw}x{ph}",
           "-r", "24", "-i", "-", *enc,
           "-timecode", "03:02:20:07", "-metadata", f"comment={BODY_TAG}", str(path)]
    subprocess.run(cmd, input=b"".join(frames), check=True)
    return path


def raw_codes(path: Path) -> list[tuple]:
    """Unique Y/Cb/Cr codes per frame straight from the file (checks the generator did not alter them)."""
    from atlas_vfx.camera_side.plate import _raw_frames, plate_ref
    ref = plate_ref(path)
    return [(tuple(np.unique(y)), tuple(np.unique(cb)), tuple(np.unique(cr)))
            for _, (y, cb, cr) in _raw_frames(ref, 0, ref["last"], False)]


def patch_means(img):
    """Mean of each half, away from the edge where chroma/scaling blends."""
    return float(img[8:-8, 8: W // 2 - 8].mean()), float(img[8:-8, W // 2 + 8: -8].mean())


def test_full(td: Path):
    p = make_plate(td / "plates" / "full.mov", "hevc", "pc")
    codes = raw_codes(p)
    want = [(tuple(sorted((c, GREY_CODE))), (512,), (512,)) for c in LEFT_CODES]
    assert codes == want, f"generator altered the codes: {codes}"
    ref = pl.plate_ref(p, first=1, last=2)
    assert ref["range"] == "full" and ref["range_source"] == "stream color_range=pc", ref["range_source"]
    assert ref["colorspace"] == "F-Log2 F-Gamut" and ref["camera_body"] == "FUJIFILM X-H2"
    assert "no recording-mode tag" in ref["colorspace_source"], ref["colorspace_source"]   # ffmpeg file: no MVTG
    assert ref["timecode"] == "03:02:20:07" and ref["frame_count"] == 4
    assert (ref["nuke_first"], ref["nuke_last"]) == (1002, 1003)
    got = list(pl.decode_rgb(ref))
    assert [i for i, _ in got] == [1, 2], [i for i, _ in got]
    worst = 0.0
    for i, img in got:
        assert img.shape == (H, W, 3)
        for want, have in zip((LEFT_CODES[i], GREY_CODE), patch_means(img)):
            worst = max(worst, abs(have * 1023 - want))
        assert float(np.max(np.abs(img[..., 0] - img[..., 1]))) < 1e-4, "neutral chroma must give R=G=B"
    assert worst < 0.1, f"full-range decode off by {worst} codes"
    stepped = [(i, round(patch_means(img)[0] * 1023)) for i, img in pl.decode_rgb(ref, 0, 3, step=2)]
    assert stepped == [(0, LEFT_CODES[0]), (2, LEFT_CODES[2])], stepped
    print(f"ok full range: HEVC 4:2:2 10-bit pc, frames 1-2 selected exactly, codes back within {worst:.3f}")
    return p, ref, got


def test_legal(td: Path, full_path: Path):
    p = make_plate(td / "plates" / "legal.mov", "prores", "tv")
    ref = pl.plate_ref(p)
    assert ref["range"] == "legal" and "tv" in ref["range_source"], ref
    img = dict(pl.decode_rgb(ref, 0, 0))[0]
    left, right = patch_means(img)
    want_l, want_r = (LEFT_CODES[0] - 64) / 876, (GREY_CODE - 64) / 876
    assert abs(left - want_l) < 2 / 1023 and abs(right - want_r) < 2 / 1023, (left, want_l, right, want_r)
    assert abs(right - GREY_CODE / 1023) > 0.005, "legal decode should not equal the full-range value"
    print(f"ok legal range: ProRes tv labelled legal, grey code {GREY_CODE} -> {right:.4f} "
          f"(full would be {GREY_CODE / 1023:.4f})")

    # the prototype's bug: a full-range file decoded as legal
    bad = pl.plate_ref(full_path, range_override="legal")
    img = dict(pl.decode_rgb(bad, 0, 0))[0]
    _, r = patch_means(img)
    assert "override" in bad["range_source"] and abs(r - GREY_CODE / 1023) > 0.005
    print(f"ok wrong-range override is recorded and visibly different ({r:.4f} vs {GREY_CODE / 1023:.4f})")


def test_colour_matrix(td: Path):
    """Known R'G'B' -> BT.709 full-range Y'CbCr written by hand here (forward direction, H.273), decoded back."""
    left, right = (0.62, 0.41, 0.18), (0.15, 0.48, 0.71)
    kr, kb = 0.2126, 0.0722

    def ycc(rgb):
        r, g, b = rgb
        y = kr * r + (1 - kr - kb) * g + kb * b
        return y, (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))

    yl, cbl, crl = ycc(left)
    yr, cbr, crr = ycc(right)
    Y = np.empty((H, W)); Y[:, : W // 2], Y[:, W // 2:] = yl * 1023, yr * 1023
    CB = np.empty((H, W // 2)); CB[:, : W // 4], CB[:, W // 4:] = 512 + cbl * 1023, 512 + cbr * 1023
    CR = np.empty((H, W // 2)); CR[:, : W // 4], CR[:, W // 4:] = 512 + crl * 1023, 512 + crr * 1023
    p = make_plate(td / "plates" / "colour.mov", "hevc", "pc", [(np.rint(Y), np.rint(CB), np.rint(CR))])
    ref = pl.plate_ref(p)
    img = next(pl.decode_rgb(ref))[1]
    worst = 0.0
    for want, sl in ((left, np.s_[8:-8, 8: W // 2 - 8]), (right, np.s_[8:-8, W // 2 + 8: -8])):
        have = img[sl].reshape(-1, 3).mean(axis=0)
        worst = max(worst, float(np.max(np.abs(have - want))) * 1023)
    # code rounding of Y, Cb, Cr is up to half a code each; through the matrix that is at most ~1.5 codes
    assert worst < 1.6, f"colour decode off by {worst:.2f} codes"
    print(f"ok colour matrix: BT.709 full-range patches back within {worst:.2f} codes (code rounding only)")


def test_flog2_grey(full_ref, got):
    one = np.full((1, 1, 3), 0.391, np.float32)
    lin = pl.to_working(one, full_ref)
    assert abs(float(lin[0, 0, 0]) - 0.18) < 1e-3 and np.allclose(lin, lin[0, 0, 0], atol=1e-5), lin
    img = got[0][1]
    _, grey = patch_means(pl.to_working(img, full_ref))
    assert abs(grey - 0.18) < 1e-3, grey
    disp = pl.to_display(img, full_ref)
    _, dgrey = patch_means(disp)
    assert 0.0 < dgrey < 1.0
    print(f"ok F-Log2 0.391 -> {float(lin[0, 0, 0]):.5f} ACEScg; decoded code {GREY_CODE} -> {grey:.5f}; "
          f"display (sRGB, ACES 2.0 SDR) grey {dgrey:.3f}")


def test_flog_grey(full_ref):
    """The original F-Log (DSCF3000 is F-Log, not F-Log2): 18 % grey is code 0.4593 (data sheet)."""
    ref = dict(full_ref, colorspace="F-Log F-Gamut")
    lin = pl.to_working(np.full((1, 1, 3), 0.4593, np.float32), ref)
    assert abs(float(lin[0, 0, 0]) - 0.18) < 1e-3 and np.allclose(lin, lin[0, 0, 0], atol=1e-5), lin
    wrong = pl.to_working(np.full((1, 1, 3), 0.4593, np.float32), full_ref)     # same code read as F-Log2
    assert float(wrong[0, 0, 0]) > 0.25, wrong
    print(f"ok F-Log 0.4593 -> {float(lin[0, 0, 0]):.5f} ACEScg (read as F-Log2 it would be {float(wrong[0, 0, 0]):.3f})")


def _fake_mov(path: Path, mode: int | None) -> Path:
    """ftyp + moov holding a FujiFilm MVTG block (LE TIFF-style records) + empty mdat. mode None: no 0x3803 record."""
    import struct

    def atom(typ, body):
        return struct.pack(">I4s", 8 + len(body), typ) + body
    recs = struct.pack("<HHII", 0x3802, 3, 1, 2)
    if mode is not None:
        recs += struct.pack("<HHII", 0x3803, 4, 1, mode)
    recs += struct.pack("<HHII", 0x3805, 4, 1, 1)
    mvtg = atom(b"MVTG", b"\x00" * 8 + recs)
    path.write_bytes(atom(b"ftyp", b"qt  \x00\x00\x00\x00") + atom(b"moov", atom(b"udta", mvtg)) + atom(b"mdat", b""))
    return path


def test_fuji_tag(td: Path):
    got = {m: pl.fuji_recording_mode(_fake_mov(td / f"fuji_{m}.mov", m)) for m in (0x10, 0x30, 0x20, 0x00)}
    assert got == {0x10: "F-Log", 0x30: "F-Log2", 0x20: "HLG", 0x00: "Normal"}, got
    assert pl.fuji_recording_mode(_fake_mov(td / "fuji_none.mov", None)) is None
    assert pl.fuji_recording_mode(td / "missing.mov") is None
    assert pl.FUJI_MODE_SPACE == {"F-Log": "F-Log F-Gamut", "F-Log2": "F-Log2 F-Gamut"}
    print("ok Fuji VideoRecordingMode tag: 0x10 F-Log, 0x30 F-Log2, 0x20 HLG, 0x00 Normal, absent -> None")


def test_config_menus():
    """Apps that read OCIO categories (Nuke's Read/Write menus) only list colour spaces tagged file-io; F-Log2 was
    missing it and did not show in Nuke 17.1 (Mike, 2026-10-19). Every active input space must carry it."""
    import PyOpenColorIO as ocio

    cfg = ocio.Config.CreateFromFile(str(pl.DEFAULT_OCIO))
    cs = cfg.getColorSpace("F-Log2 F-Gamut")
    assert "file-io" in list(cs.getCategories()), list(cs.getCategories())
    missing = [c.getName() for c in cfg.getColorSpaces(ocio.SEARCH_REFERENCE_SPACE_ALL, ocio.COLORSPACE_ACTIVE)
               if c.getFamily().startswith("Input") and "file-io" not in list(c.getCategories())]
    assert not missing, missing
    print("ok config: F-Log2 F-Gamut and every other input space are tagged file-io (shown in Nuke's Read menu)")


def test_tracking(td: Path, full_ref: dict):
    out, kept = pl.write_tracking_proxies(full_ref, td / "track", width=128, keep_every=1)
    files = sorted(f.name for f in out.glob("plate_*.png"))
    assert files == ["plate_0001.png", "plate_0002.png"], files
    side = json.loads((out / "track_proxy.json").read_text(encoding="utf-8"))
    assert side["node"] == "AtlasVFXTrackingProxy" and side["outputs"]["size"] == [128, 64]
    assert side["inputs"]["plate_ref"]["quick_sha256"] == full_ref["quick_sha256"]
    assert len(kept) == 2 and kept[0].shape == (64, 128) and kept[0].dtype == np.uint8
    # left dark, right brighter in frame 1 (300 < 400), and the stretch uses the full 8-bit range
    assert kept[0][:, :40].mean() < kept[0][:, -40:].mean() and kept[0].max() > 200
    try:
        pl.write_tracking_proxies(full_ref, Path(full_ref["path"]).parent / "x", width=64)
    except ValueError as e:                     # resolve_out_dir: the plate's folder is refused
        assert "plate" in str(e), e
    else:
        raise AssertionError("wrote proxies into the plate folder")
    print(f"ok tracking proxies: {files}, sidecar, refuses the plate folder")


def test_real():
    if not REAL.is_file():
        print(f"skip real plate: {REAL} not found")
        return
    ref = pl.plate_ref(REAL, first=129, last=296)
    # DSCF3000 is F-Log (Mike, 2026-10-20; FujiFilm tag 0x3803 = 0x10), read from the file, not the body default
    assert "VideoRecordingMode F-Log " in ref["colorspace_source"], ref["colorspace_source"]
    want = {"range": "full", "camera_body": "FUJIFILM X-H2", "colorspace": "F-Log F-Gamut", "width": 4096,
            "height": 2160, "frame_count": 528, "timecode": "03:02:20:07", "fps": [24, 1],
            "nuke_first": 1130, "nuke_last": 1297}
    bad = {k: (ref[k], v) for k, v in want.items() if ref[k] != v}
    assert not bad, bad
    i, img = next(pl.decode_rgb(ref, 129, 129, width=256))
    assert i == 129 and img.shape == (136, 256, 3) and 0.05 < float(img.mean()) < 0.95
    print(f"ok real plate: DSCF3000 {ref['range']} ({ref['range_source']}), {ref['colorspace']} "
          f"({ref['colorspace_source']}), stream tags {ref['stream_color_tags']['color_transfer']} ignored, "
          f"S2-S3 = Nuke {ref['nuke_first']}-{ref['nuke_last']}, frame 129 mean code {img.mean():.3f}")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        full_path, full_ref, got = test_full(td)
        test_legal(td, full_path)
        test_colour_matrix(td)
        test_flog2_grey(full_ref, got)
        test_flog_grey(full_ref)
        test_fuji_tag(td)
        test_config_menus()
        test_tracking(td, full_ref)
    test_real()
    return 0


if __name__ == "__main__":
    sys.exit(main())
