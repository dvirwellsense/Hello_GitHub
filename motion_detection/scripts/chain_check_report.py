"""Report for chain_check.py (L=4 chained second check) next to the adjacent-pair pipeline.

  python scripts/chain_check_report.py results/chain4_2026_10 > results/chain4_2026_10/report.txt

Uses results/chain4_2026_10/score/<scan>.csv (real runs, no injection), .../inject/<scan>.csv (synthetic drift / step),
results/k16_2026_10 (adjacent-pair run, k=16, 0.5 px) and results/injection_2026_10 (adjacent-pair injection).
A pair i-4 -> i spans adjacent pairs i-3 .. i: moving when any of them is a labelled moving pair, negative when all
of them are negatives (compare_mask_runs.py definition). Alarms: persistence 2 on each check separately; the combined
alarm is adjacent-pair alarm OR chained alarm. Thresholds of the chained check are candidates, nothing is applied.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_size_report as c  # noqa: E402
import compare_mask_runs as cm  # noqa: E402

L = 4
THRS = (0.8, 1.0, 1.2, 1.4, 1.6)


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def negatives(n, frames):
    near = set()
    for k in c.MOVING.get(n, set()) | c.LABELLED.get(n, set()) | c.UNVERIFIED.get(n, set()):
        near |= set(range(k - 2, k + 3))
    return {k for k in frames if n in c.STATIC or k in c.PAUSE.get(n, set()) or k not in near}


def main(root, score_dir=None, inject_dir=None, adj_injection='results/injection_2026_10', real='results/k16_2026_10'):
    root = Path(root)
    score_dir = Path(score_dir) if score_dir else root / 'score'
    inject_dir = Path(inject_dir) if inject_dir else root / 'inject'
    sc = {p.stem: {int(r['frame']): fl(r['chain_cl16']) for r in csv.DictReader(open(p))} for p in sorted(score_dir.glob('*.csv'))}
    print(f'== Chained L={L}, no injection: score per scan (px)')
    for n, d in sc.items():
        v = np.array(list(d.values()))
        f = v[np.isfinite(v)]
        print(f'  {n:12s} {len(f):3d}/{len(v):3d} scored  median {np.median(f):.2f}  max {f.max():.2f}')
    st = np.concatenate([[x for x in sc[n].values() if np.isfinite(x)] for n in list(c.STATIC) + ['phantom'] if n in sc])
    print(f'  static scans + phantom: median {np.median(st):.2f}, 99th pct {np.percentile(st, 99):.2f}, max {st.max():.2f}\n')

    inj = []
    for p in sorted(inject_dir.glob('*.csv')):
        inj += list(csv.DictReader(open(p)))
    g = defaultdict(list)
    for r in inj:
        if r['kind'] != 'none':
            g[(r['kind'], r['region'], int(r['angle_deg']), float(r['size_mm']))].append(fl(r['chain_cl16']))
    # adjacent-pair detection of the same per-frame displacement (injection_2026_10, k=16, 0.5 px)
    adj = defaultdict(list)
    for p in Path(adj_injection).glob('*.csv'):
        for r in csv.DictReader(open(p)):
            if r['region'] in ('r80', 'body'):
                adj[(r['region'], int(r['angle_deg']), float(r['size_mm']))].append(np.nan_to_num(fl(r['epi_enc_cl16'])) > 0.5)
    print(f'== Synthetic motion on the static scans ({len({(r["frame"], r["chain_cl16"]) for r in inj if r["kind"] == "none"})} pairs): share detected')
    print('   drift v mm/frame: adjacent pair sees v per pair (share of adjacent pairs above 0.5 px, injection_2026_10);')
    print('   chained L=4 sees 4v. step s mm: the adjacent pair that contains the step sees s, but only that one pair.')
    for kind in ('drift', 'step'):
        for region in ('r80', 'body'):
            for ang in (0, 60):
                sizes = sorted({k[3] for k in g if k[:3] == (kind, region, ang)})
                if not sizes:
                    continue
                print(f'  {kind:5s} {region:4s} {ang:2d} deg  size  ' + '  '.join(f'{s:5g}' for s in sizes))
                if kind == 'drift':
                    print('      adjacent pair (0.5 px)  ' + '  '.join(f'{np.mean(adj[(region, ang, s)]):5.2f}' if adj.get((region, ang, s)) else '    -' for s in sizes))
                for t in THRS:
                    print(f'      chained > {t:.1f} px       ' + '  '.join(f'{np.mean(np.nan_to_num(np.array(g[(kind, region, ang, s)])) > t):5.2f}' for s in sizes))
        print()

    print('== Real scans: alarms (persistence 2). Adjacent-pair pipeline as now, chained check at candidate thresholds')
    adj_al, adj_neg = {}, {}
    for n in cm.SCANS:
        R = cm.load(real, n)
        if R is None:
            continue
        s1 = {int(r['frame']): fl(r['epi_enc_cl16']) for r in R}
        adj_al[n] = set(cm.alarms({k for k, v in s1.items() if v >= 0.5}))
        adj_neg[n] = negatives(n, set(s1) | set(range(min(s1) - L, min(s1))))
    for t in THRS:
        new_neg, new_mov, tot_neg, tot_mov, per = [], 0, 0, 0, {}
        for n in cm.SCANS:
            if n not in sc or n not in adj_al:
                continue
            d = sc[n]
            al = set(cm.alarms({k for k, v in d.items() if np.isfinite(v) and v > t}))
            neg = {k for k in d if all(m in adj_neg[n] for m in range(k - L + 1, k + 1))}
            mov = {k for k in d if any(m in c.MOVING.get(n, set()) for m in range(k - L + 1, k + 1))}
            tot_neg += len(neg)
            tot_mov += len(mov)
            new_neg += [(n, k) for k in sorted(al & neg - adj_al[n])]
            new_mov += len(al & mov)
            extra = sorted(al - adj_al[n])
            if extra:
                per[n] = extra
        print(f'  chained > {t:.1f} px: chained alarms on negative spans {len(new_neg)}/{tot_neg} {new_neg}; '
              f'chained alarms on moving spans {new_mov}/{tot_mov}')
        for n, e in per.items():
            print(f'      {n:12s} alarm frames added to the adjacent-pair alarms: {e}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='results/chain4_2026_10')
    ap.add_argument('--score-dir', help='CSVs with a chain_cl16 column (default <root>/score; evaluate_scan.py runs work too)')
    ap.add_argument('--inject-dir', help='chain_check.py inject CSVs (default <root>/inject)')
    ap.add_argument('--adjacent-injection', default='results/injection_2026_10')
    ap.add_argument('--real', default='results/k16_2026_10', help='adjacent-pair runs (epi_enc_cl16)')
    a = ap.parse_args()
    main(a.root, a.score_dir, a.inject_dir, a.adjacent_injection, a.real)
