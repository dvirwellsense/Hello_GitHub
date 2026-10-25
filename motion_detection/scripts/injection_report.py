"""Summary of inject_motion.py runs: detection rate by size, direction and region, and what the threshold costs.

  python scripts/injection_report.py results/injection_2026_10/*.csv --real results/k16_2026_10 > report.txt

Detection = cluster score above the threshold for one pair (persistence 1). The alarm also needs the next pair
(persistence 2), so a motion that is over within one frame interval is reported per pair only.
False-alarm side: the real runs of all scans (no injection), negatives and moving pairs as in compare_mask_runs.py.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_size_report as c  # noqa: E402
import compare_mask_runs as cm  # noqa: E402

G = {}

def fl(v):
    return float(v) if v not in ('', 'nan', None) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--real', default='results/k16_2026_10', help='folder with the real-run CSVs of the static scans')
    ap.add_argument('--threshold', type=float, default=0.5)
    a = ap.parse_args()
    rows = []
    for p in a.csv:
        for r in csv.DictReader(open(p)):
            r['scan'] = Path(p).stem
            rows.append(r)
    scans = sorted({r['scan'] for r in rows})
    npairs = len({(r['scan'], r['frame']) for r in rows})
    print(f'Scans: {", ".join(scans)}; {npairs} pairs, {len(rows)} scored pairs')
    ctl = [fl(r['epi_enc_cl16']) for r in rows if r['region'] == 'none']
    print(f'Control (no injection) k=16: median {np.nanmedian(ctl):.3f} px, max {np.nanmax(ctl):.3f} px')
    print('Body points above the detector (median per pair) and r40 detection at 5 mm across the line, k=8/16/30:')
    for s in scans:
        R = [r for r in rows if r['scan'] == s]
        m = [r for r in R if r['region'] == 'r40' and float(r['size_mm']) == 5.0 and int(r['angle_deg']) == 0]
        det = [np.mean([np.nan_to_num(fl(r[c])) > a.threshold for r in m]) for c in ('epi_enc_cl', 'epi_enc_cl16', 'epi_enc_cl30')]
        print(f'  {s:12s} n_body {np.median([int(r["n_body"]) for r in R if r["region"] == "none"]):6.0f}   ' + ' / '.join(f'{d:.2f}' for d in det))
    print()

    for col, k in (('epi_enc_cl16', 16), ('epi_enc_cl', 8), ('epi_enc_cl30', 30)):
        g = defaultdict(list)
        for r in rows:
            if r['region'] != 'none':
                g[(r['region'], float(r['size_mm']), int(r['angle_deg']))].append(fl(r[col]))
        print(f'== k={k}: share of pairs detected (score > {a.threshold} px); unmeasurable counts as missed')
        angles = sorted({x[2] for x in g})
        for region in ('r40', 'r80', 'body'):
            print(f'  region {region:5s} size_mm ' + ' '.join(f'{x:>5d}deg' for x in angles))
            for size in sorted({x[1] for x in g if x[0] == region}):
                cells = []
                for ang in angles:
                    v = np.array(g[(region, size, ang)])
                    cells.append(f'{np.mean(np.nan_to_num(v, nan=0) > a.threshold):8.2f}')
                print(f'                {size:7.2f} ' + ' '.join(cells))
        if k == 16:
            print('\n  median score (px) across the line (0 deg), region r40: ' + ', '.join(
                f'{s:g} mm -> {np.nanmedian(g[("r40", s, 0)]):.2f}' for s in sorted({x[1] for x in g if x[0] == 'r40'})))
        print()

    print('== Threshold trade-off (k=16), real runs, no injection. Negatives and moving pairs as in compare_mask_runs.py')
    print('   (static scans, confirmed pauses, unlabelled pairs more than 2 pairs from a label)')
    g16 = defaultdict(list)
    for r in rows:
        if r['region'] == 'r80' and int(r['angle_deg']) == 0:
            g16[float(r['size_mm'])].append(fl(r['epi_enc_cl16']))
    neg, mov = {}, {}
    for n in cm.SCANS:
        R = cm.load(a.real, n)
        if R is None:
            continue
        sc = {int(r['frame']): fl(r['epi_enc_cl16']) for r in R}
        M = c.MOVING.get(n, set())
        near = set()
        for k in M | c.LABELLED.get(n, set()) | c.UNVERIFIED.get(n, set()):
            near |= set(range(k - 2, k + 3))
        neg[n] = [k for k in sc if n in c.STATIC or k in c.PAUSE.get(n, set()) or k not in near]
        mov[n] = sorted(M)
        G[n] = sc
    nneg = sum(len(v) for v in neg.values())
    nmov = sum(len(v) for v in mov.values())
    for thr in (0.5, 0.4, 0.35, 0.3, 0.25):
        fp = [(n, k) for n in neg for k in neg[n] if np.nan_to_num(G[n][k]) > thr]
        sus = {n: {k for k, v in G[n].items() if np.nan_to_num(v) > thr} for n in G}
        fa = [(n, k) for n in neg for k in cm.alarms(sus[n]) if k in neg[n]]
        tp = sum(len(sus[n] & set(mov[n])) for n in mov)
        det = ', '.join(f'{s:g} mm {np.mean(np.nan_to_num(np.array(g16[s])) > thr):.2f}' for s in (0.5, 0.75, 1.0, 1.5))
        print(f'  threshold {thr:.2f} px: labelled moving pairs flagged {tp}/{nmov}; negatives flagged {len(fp)}/{nneg}; '
              f'negative alarm frames (persistence 2) {len(fa)} {fa}; injected r80 across the line detected: {det}')

if __name__ == '__main__':
    main()
