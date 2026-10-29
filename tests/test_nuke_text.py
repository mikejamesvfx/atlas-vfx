"""Nuke text is built from data, never pasted (spec panel 2026-10-26, Vulns 1-3 and item 3).

- nuke_text.py_literal: any data round-trips through the generated Python as data, one statement, nothing more
- nuke_text.nk_str / nk_name: a quoted .nk knob value cannot end early or carry a TCL [..] / $VAR; node names are \\w only
- plate.safe_name: one file-name component, refused (not repaired) when it is anything else
- plate.resolve_out_dir: every node output resolves under the work root(s), never the repo, never next to the plate
- the writers on them: Nuke Builder (Python), Sensor RS / Lens, Plate Lock Write and Cross-View (.nk) keep exactly
  the nodes they meant to write when a path or name is hostile
- a node refuses an out_dir outside the roots and a name that is not one path component

Run: <V141 python> tests\\test_nuke_text.py   (plain script, exits non-zero on failure; no pytest)
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TMP = Path(tempfile.mkdtemp(prefix="avfx_nuke_text_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)          # the one place tests may write; read at call time

from atlas_vfx.camera_side import nuke_text as nt  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.deliver import nuke_builder as nb  # noqa: E402
from atlas_vfx.platelock import crossview_job as cvj  # noqa: E402
from atlas_vfx.platelock.core import deliver as pdl  # noqa: E402
from atlas_vfx.sensor import maps_io as mio  # noqa: E402

# breaks out of r'''...''' and keeps json.loads happy so the statement after it runs (the 2026-10-26 proof)
PY_HOSTILE = "S\"}''') if 0 else 0 ; INJECTED = 1 ; _ = (r'''"
# closes a .nk file knob and node, adds a node whose onCreate runs Python, reopens a Read
NK_HOSTILE = 'x.####.exr"\n}\nNoOp {\n onCreate "python: import os"\n name a\n}\nRead {\n file "y'


def raises(exc, fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{a!r} should raise {exc.__name__}")


def nk_unquote(q: str) -> str:
    """Inverse of nk_str, written here so the test does not trust the module's own decoder."""
    assert len(q) >= 2 and q[0] == '"' and q[-1] == '"', q
    body, out, i = q[1:-1], [], 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            n = body[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t"}.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def nk_nodes(text: str) -> list[str]:
    return re.findall(r"^(\w+) \{$", text, re.M)


def knob(text: str, name: str) -> list[str]:
    return [line.strip()[len(name) + 1:] for line in text.splitlines() if line.strip().startswith(name + " ")]


# ------------------------------------------------------------------------------------------------- helpers

def test_py_literal_round_trips_anything():
    data = {"shot": PY_HOSTILE, "path": "C:/x'''y/\\z\n", "text": "\u2028 line sep, \u00e9, \"q\" 'q'",
            "n": [1, 2.5, None, True, {"k": "'''"}]}
    lit = nt.py_literal(data)
    assert "\n" not in lit, "one line, so it can never end a comment or start a statement"
    src = "import json\nDATA = " + lit + "\n"
    assert len(ast.parse(src).body) == 2, ast.dump(ast.parse(src))
    ns: dict = {}
    exec(compile(src, "<py_literal>", "exec"), ns)
    assert ns["DATA"] == data and "INJECTED" not in ns
    print("ok py_literal: hostile strings round-trip as data in one statement")


def test_nk_str_cannot_end_early():
    for s in (NK_HOSTILE, 'C:/a "b"/c.####.exr', "[python {import os}]", "$HOME/x", "a\\b{c}d", "plain/path.exr",
              "tab\there", "cr\rhere"):
        q = nt.nk_str(s)
        assert q[0] == '"' and q[-1] == '"' and "\n" not in q and "\r" not in q, q
        assert nk_unquote(q) == s, (s, q)
        body = q[1:-1]
        for ch in '"[]${}':
            assert all(body[i - 1] == "\\" for i, c in enumerate(body) if c == ch and i), (ch, q)
    raises(ValueError, nt.nk_str, "nul\x00byte")
    raises(ValueError, nt.nk_str, "esc\x1b[31m")
    print("ok nk_str: quote, backslash, [ ] $ { } and line breaks escaped; control characters refused")


def test_nk_name_is_word_only():
    assert nt.nk_name("Read_a") == "Read_a"
    assert nt.nk_name("Read_a b") == "Read_a_b"
    assert nt.nk_name("1x") == "_1x"
    assert nt.nk_name('a\nb"c}') == "a_b_c_"
    raises(ValueError, nt.nk_name, "")
    print("ok nk_name: anything outside [A-Za-z0-9_] becomes _, leading digit prefixed, empty refused")


def test_safe_name():
    for good in ("DSCF3000_S2S3", "grass near.v2", "\u00e9lan", "AVFX_Camera", "a-b"):
        assert pl.safe_name(good) == good, good
    assert pl.safe_name("  padded ") == "padded"
    for bad in ("", "..", ".", "a/b", "a\\b", PY_HOSTILE, "x\ny", ".hidden", "name.", "a:b", "a*b", "a?b", 'a"b',
                "a'b", "a|b", "..\\syn"):
        raises(ValueError, pl.safe_name, bad)
    print("ok safe_name: one path component, letters digits space _ . - only; traversal and quotes refused")


def test_resolve_out_dir():
    plate = TMP / "plates" / "p.MOV"
    d = pl.resolve_out_dir("", TMP / "default")
    assert d == (TMP / "default").resolve(), d
    d = pl.resolve_out_dir(f'"{TMP / "quoted"}"', TMP)
    assert d == (TMP / "quoted").resolve(), d
    d = pl.resolve_out_dir(str(TMP / "work"), TMP, plate_path=plate)
    assert d == (TMP / "work").resolve()
    msg = raises(ValueError, pl.resolve_out_dir, str(ROOT / "out"), TMP)
    assert "repo" in msg, msg
    msg = raises(ValueError, pl.resolve_out_dir, str(TMP / ".." / "escape"), TMP)
    assert "work root" in msg, msg
    # an absolute folder outside every work root on this OS (a Windows path is relative on Linux: it fell in the repo)
    outside = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp" / "avfx" if os.name == "nt" else Path("/var/tmp/avfx")
    msg = raises(ValueError, pl.resolve_out_dir, str(outside), TMP)
    assert "work root" in msg, msg
    msg = raises(ValueError, pl.resolve_out_dir, "", ROOT / "default_in_repo")
    assert "repo" in msg, msg
    msg = raises(ValueError, pl.resolve_out_dir, str(TMP / "plates" / "proxies"), TMP, plate_path=plate)
    assert "plate" in msg, msg
    msg = raises(ValueError, pl.resolve_out_dir, str(TMP / "plates"), TMP, plate_path=plate)
    assert "plate" in msg, msg
    roots = pl.out_roots()
    assert any(r == Path(os.path.normcase(str(TMP.resolve()))) for r in roots), roots
    print("ok resolve_out_dir: default and quoted values accepted under the roots; repo, escape via .., a folder "
          "outside the roots and the plate's folder refused")


# ------------------------------------------------------------------------------------------------- writers

def builder_data(**over) -> dict:
    kw = dict(shot="SYN_S1", plate=nb.plate_source(r"C:\plates\SYN.MOV", "dpx"), plate_colorspace="F-Log F-Gamut",
              plate_frames=(1130, 1297), fmt=(4096, 2160, 1.0), ocio_config="C:/cfg/a.ocio",
              camera_chan="C:/cam/a.chan",
              camera_settings={"rot_order": "ZXY", "haperture": 23.5, "vaperture": 12.39, "win_translate": [0, 0]},
              layers=[{"name": "grass", "pattern": "C:/l/grass.%04d.exr", "first": 1130, "last": 1297,
                       "colorspace": "ACEScg", "premultiplied": True}],
              write_pattern="C:/out/c.%04d.exr", prefix="AVFX_")
    kw.update(over)
    return nb.build_data(**kw)


def test_builder_data_never_becomes_code():
    clean = nb.script_text(builder_data())
    hostile_path = "C:/x'''y\n/\"a\".ocio"
    hostile = nb.script_text(builder_data(ocio_config=hostile_path))
    assert len(ast.parse(clean).body) == len(ast.parse(hostile).body), "a hostile path added a statement"
    head = hostile[: hostile.index("\nimport os\nimport nuke")]
    ns: dict = {}
    exec(compile(head, "build_head.py", "exec"), ns)
    assert ns["DATA"]["ocio_config"] == hostile_path and "INJECTED" not in ns
    for line in clean.splitlines():
        if line.startswith("#"):
            assert "SYN_S1" not in line and "AVFX_" not in line, f"data in a comment: {line}"
    # the names that become node names or file names are refused when they are not plain
    raises(ValueError, builder_data, shot="S\nX")
    raises(ValueError, builder_data, shot=PY_HOSTILE)
    raises(ValueError, builder_data, shot="..\\S")
    raises(ValueError, builder_data, prefix="A B_")
    raises(ValueError, builder_data, prefix="P'")
    raises(ValueError, builder_data, prefix="")           # would delete every node in the Nuke script on re-run
    bad_layer = [{"name": PY_HOSTILE, "pattern": "C:/l/x.%04d.exr", "first": 1, "last": 2, "colorspace": "ACEScg",
                  "premultiplied": True}]
    raises(ValueError, builder_data, layers=bad_layer)
    print("ok nuke builder: hostile path round-trips as data (same statement count, no comment carries data); "
          "shot, prefix and layer names refused when not plain")


def test_sensor_nk_keep_their_nodes():
    hostile_dir = TMP / ('maps"\n}\nNoOp {\n onCreate "python: import os"\n name a\n}\nRead {\n file "y')
    for make in (lambda d: mio.nuke_rs(d, 1130, 1140, blur=2), mio.nuke_lens):
        clean, bad = make(TMP / "maps"), make(hostile_dir)
        assert nk_nodes(clean) == nk_nodes(bad) and not knob(bad, "onCreate"), (nk_nodes(clean), nk_nodes(bad))
        files = knob(bad, "file")
        assert files and all(nk_unquote(f).startswith(str(hostile_dir).replace("\\", "/")) for f in files), files
        assert all(" " not in n for n in knob(bad, "name")), knob(bad, "name")
    print("ok sensor .nk: rs and lens scripts keep their node list with a hostile folder; file knobs round-trip")


def test_platelock_nk_keep_their_nodes():
    shot = {"first_frame": 1001, "last_frame": 1003, "ai_first_frame": 1001, "ai_path": NK_HOSTILE,
            "people": {"le ft": {"source_frames": [1001, 1001, 1002]}}}
    text = pdl.nuke_script(shot, str(TMP / 'lock"\n}'))
    assert nk_nodes(text) == ["Root", "Read", "Read", "Read", "Read", "TimeWarp", "Read", "Read"], nk_nodes(text)
    assert not knob(text, "onCreate"), "the hostile value opened a knob of its own"
    ai = [f for f in knob(text, "file") if nk_unquote(f) == NK_HOSTILE.replace("\\", "/")]
    assert ai, knob(text, "file")
    assert "Read_AI_le_ft" in knob(text, "name") and "TimeWarp_le_ft" in knob(text, "name"), knob(text, "name")
    labels = knob(text, "label")
    assert labels and all(nk_unquote(lab) for lab in labels)

    cv = cvj._nuke(TMP / 'cv"\n}', ["id_a", "id-b"], [1001, 1002], NK_HOSTILE)
    assert nk_nodes(cv) == ["Root", "Read", "Read", "Read", "Read", "Read", "Read"], nk_nodes(cv)
    assert not knob(cv, "onCreate") and "Read_id_b_stmap" in knob(cv, "name"), knob(cv, "name")
    print("ok plate lock .nk: lock and cross-view scripts keep their node list with a hostile ai_path, out_dir "
          "and performer name")


def test_write_lock_refuses_traversal_in_a_person_name():
    msg = raises(ValueError, pdl.check_people_names, ["left", "..\\right"])
    assert "right" in msg, msg
    assert pdl.check_people_names(["left", "ri ght"]) == ["left", "ri ght"]
    print("ok write_lock: a performer name that is not one path component is refused before any folder is made")


# ------------------------------------------------------------------------------------------------- a node

def test_node_refuses_out_dir_and_name():
    from test_depth_set_proxy import camera

    from atlas_vfx.camera_side.node_camera_export import AtlasVFXCameraExport
    cam = camera()
    node = AtlasVFXCameraExport()
    msg = raises(ValueError, node.run, cam.data, 1.0, "syn", str(ROOT / "out"))
    assert "repo" in msg, msg
    msg = raises(ValueError, node.run, cam.data, 1.0, "syn", str(TMP.parent / "avfx_escape"))
    assert "work root" in msg, msg
    msg = raises(ValueError, node.run, cam.data, 1.0, "..\\syn", str(TMP / "ok"))
    assert "name" in msg, msg
    assert not (TMP.parent / "avfx_escape").exists() and not (ROOT / "out").exists()
    res = node.run(cam.data, 1.0, "syn", str(TMP / "ok"))
    chan = Path(res["result"][0])
    assert chan.is_file() and (TMP / "ok").resolve() in chan.resolve().parents, chan
    print("ok node: Camera Export refuses the repo, a folder outside the roots and a traversal name; writes under "
          "the allowed root")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
        except Exception:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - len(failed)}/{len(tests)} passed" + (f", failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
