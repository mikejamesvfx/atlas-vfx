# Run inside Nuke (Script Editor): exec(open(r'C:/Users/miike/howling-yowies/nuke/export_for_claude.py').read())
# Writes C:/Users/miike/howling-yowies/nuke/export/ :
#   <camera>.chan       frame tx ty tz rx ry rz focal (Nuke chan layout, plus focal)
#   <camera>.json       per-frame translate/rotate/focal/apertures/win_translate + rot/xform order
#   cameratracker_<name>.json   the CameraTracker's settings and solve stats (small knobs only)
#   pointcloud_<name>.obj       the solved point cloud, if a CameraTrackerPointCloud node exists (or can be made)
#   summary.txt         what was found
# Read-only on the script: nothing is changed except a temporary WriteGeo node, deleted again.
import json, os
import nuke

OUT = r'C:/Users/miike/howling-yowies/nuke/export'
if not os.path.isdir(OUT):
    os.makedirs(OUT)
root = nuke.root()
first, last = int(root['first_frame'].value()), int(root['last_frame'].value())
nodes = nuke.allNodes(recurseGroups=True)
log = ['script: %s' % root.name(), 'frame range: %d-%d' % (first, last)]


def kv(node, name, frame=None, idx=None):
    k = node.knob(name)
    if k is None:
        return None
    try:
        if idx is None:
            return k.valueAt(frame) if frame is not None else k.value()
        return k.valueAt(frame, idx) if frame is not None else k.value(idx)
    except Exception:
        try:
            return k.value()
        except Exception:
            return None


cams = [n for n in nodes if n.Class() in ('Camera', 'Camera2', 'Camera3', 'Camera4')]
for c in cams:
    rows, chan = [], []
    for f in range(first, last + 1):
        t = [kv(c, 'translate', f, i) for i in range(3)]
        r = [kv(c, 'rotate', f, i) for i in range(3)]
        foc = kv(c, 'focal', f)
        rows.append({'frame': f, 'translate': t, 'rotate': r, 'focal': foc,
                     'haperture': kv(c, 'haperture', f), 'vaperture': kv(c, 'vaperture', f),
                     'win_translate': [kv(c, 'win_translate', f, 0), kv(c, 'win_translate', f, 1)]})
        chan.append('%d\t%s\t%s\t%s' % (f, '\t'.join('%.6f' % v for v in t), '\t'.join('%.6f' % v for v in r), '%.6f' % foc))
    meta = {'name': c.name(), 'class': c.Class(), 'rot_order': c['rot_order'].value() if c.knob('rot_order') else None,
            'xform_order': c['xform_order'].value() if c.knob('xform_order') else None,
            'animated_translate': c['translate'].isAnimated() if c.knob('translate') else None, 'frames': rows}
    json.dump(meta, open(os.path.join(OUT, '%s.json' % c.name()), 'w'), indent=1)
    open(os.path.join(OUT, '%s.chan' % c.name()), 'w').write('\n'.join(chan) + '\n')
    log.append('camera %s (%s): rot_order %s, focal@first %.4f, haperture %.4f, animated %s' % (
        c.name(), c.Class(), meta['rot_order'], rows[0]['focal'] or 0, rows[0]['haperture'] or 0, meta['animated_translate']))

for t in [n for n in nodes if n.Class() == 'CameraTracker']:
    d = {}
    for name, k in t.knobs().items():
        try:
            s = k.toScript()
        except Exception:
            continue
        if s is not None and len(s) < 400:
            d[name] = s
    json.dump(d, open(os.path.join(OUT, 'cameratracker_%s.json' % t.name()), 'w'), indent=1, sort_keys=True)
    picks = {k: d[k] for k in d if any(w in k.lower() for w in ('error', 'focal', 'film', 'lens', 'distort', 'range', 'solve', 'aperture', 'camera'))}
    log.append('CameraTracker %s: %d knobs saved; %s' % (t.name(), len(d), json.dumps(picks)[:1500]))

pcs = [n for n in nodes if 'PointCloud' in n.Class()]
for pc in pcs:
    try:
        wg = nuke.nodes.WriteGeo(file=os.path.join(OUT, 'pointcloud_%s.obj' % pc.name()).replace('\\', '/'))
        wg.setInput(0, pc)
        nuke.execute(wg, first, first)
        nuke.delete(wg)
        log.append('point cloud %s -> obj' % pc.name())
    except Exception as e:
        log.append('point cloud %s not written: %s' % (pc.name(), e))
if not pcs:
    log.append('no point cloud node: in the CameraTracker use Export > Scene+ (or Point Cloud), then run this again')
for n in [n for n in nodes if n.Class() in ('LensDistortion', 'LensDistortion2')]:
    log.append('LensDistortion %s: %s' % (n.name(), json.dumps({k: n[k].toScript() for k in n.knobs() if any(w in k.lower() for w in ('distortion', 'model', 'focal', 'film', 'centre', 'center', 'denominator', 'numerator'))})[:1200]))
open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
print('export written to', OUT)
