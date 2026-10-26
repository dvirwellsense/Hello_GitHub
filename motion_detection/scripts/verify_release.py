"""Compare a local re-run with the runs of the release: same pairs, same scores, same alarms.

  python scripts/verify_release.py <release_runs_dir> <new_runs_dir>

For every run folder present in both (regular, combined, mediapipe_*, yolov8n640_every4, realtime_*, injection,
chain_injection) and every CSV in it: rows are matched by frame (and by region/size/angle for the injections);
scores are compared with a tolerance of 1e-9 px, flags exactly. Timing columns (t_*_ms) are not compared.
"""
import csv
import sys
from pathlib import Path

SCORES = ('epi_enc_cl16', 'static_cl', 'chain_cl16', 'n_body', 'n_chain')
FLAGS = ('exposure', 'same_angle', 'motion_state', 'alarm', 'alarm_adjacent', 'alarm_chain', 'chain_suspect', 'suspect')


def key(r):
    return tuple(r.get(k, '') for k in ('frame', 'kind', 'region', 'size_mm', 'angle_deg'))


def fl(v):
    """Number, or None for empty / nan (an empty cell and 'nan' both mean 'not measured')."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def main(ref_root, new_root):
    ref_root, new_root = Path(ref_root), Path(new_root)
    total_bad = 0
    for d in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        nd = new_root / d.name
        if not nd.is_dir():
            continue
        for f in sorted(d.glob('*.csv')):
            g = nd / f.name
            if not g.exists():
                print(f'{d.name}/{f.name}: missing in the new run')
                total_bad += 1
                continue
            a = {key(r): r for r in csv.DictReader(open(f))}
            b = {key(r): r for r in csv.DictReader(open(g))}
            bad = []
            if set(a) != set(b):
                bad.append(f'rows differ: {len(set(a) ^ set(b))}')
            for k in set(a) & set(b):
                for c in a[k]:
                    if c.startswith('t_') or c not in b[k]:
                        continue
                    x, y = a[k][c], b[k][c]
                    if c in FLAGS or c in ('mask_computed', 'mask_from_frame'):
                        if x != y:
                            bad.append((k[0], c, x, y))
                    elif c in SCORES or c.startswith('epi_') or c.endswith('_px'):
                        fx, fy = fl(x), fl(y)
                        if (fx is None) != (fy is None) or (fx is not None and abs(fx - fy) > 1e-9):
                            bad.append((k[0], c, x, y))
            total_bad += len(bad)
            print(f'{d.name}/{f.name}: {"identical" if not bad else f"{len(bad)} differences, e.g. {bad[:3]}"}')
    print('ALL IDENTICAL' if not total_bad else f'{total_bad} differences in total')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
