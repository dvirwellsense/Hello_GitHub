"""Summarize results/cluster_size_*/<scan>.csv (from cluster_size_study.py): for each cluster size k, the noise on
negatives, the zero-false-alarm threshold, and the detection of moving pairs at that threshold and at 0.5 px."""
import csv
import sys
from pathlib import Path

import numpy as np

KS = (8, 12, 16, 20, 30, 45)
R = lambda a, b: set(range(a, b + 1))
# verified: labeller confirmed which labelled pairs move / pause
MOVING = {'A30': R(55, 64) | R(68, 69),
          'A44': R(56, 57) | R(59, 61) | R(63, 68) | R(70, 72) | R(75, 76) | {80} | R(83, 85)}
PAUSE = {'A30': R(65, 67), 'A44': {58, 62, 69, 73, 74, 77, 78, 81, 82}}
# labelled but pauses not checked: excluded from negatives, reported separately
LABELLED = {'A08': R(74, 86), 'A12': R(57, 61), 'A26': R(59, 68), 'A28_1': R(63, 86)}
UNVERIFIED = {'A28_1': R(106, 110)}  # suspects outside the label, not verified
STATIC = {'A07', 'A11', 'legs_static', 'VC01_bh'}


def load(d, name):
    return {int(r['frame']): {k: float(r[k]) for k in r if k != 'frame'} for r in csv.DictReader(open(Path(d) / f'{name}.csv'))}


def main(d):
    names = ['A30', 'A44', 'A08', 'A12', 'A26', 'A28_1', 'A07', 'A11', 'legs_static', 'VC01_bh']
    data = {n: load(d, n) for n in names}
    neg, pos, lab = {k: [] for k in KS}, {k: [] for k in KS}, {n: {k: [] for k in KS} for n in LABELLED}
    for n in names:
        marked = (MOVING.get(n, set()) | LABELLED.get(n, set()) | UNVERIFIED.get(n, set()))
        near = set()
        for f in marked:
            near |= {f - 2, f - 1, f, f + 1, f + 2}
        for f, row in data[n].items():
            for k in KS:
                v = row[f'k{k}']
                if np.isnan(v):
                    continue
                if n in STATIC or f in PAUSE.get(n, set()) or (f not in near and n not in STATIC):
                    neg[k].append(v)
                elif f in MOVING.get(n, set()):
                    pos[k].append(v)
                if n in LABELLED and f in LABELLED[n]:
                    lab[n][k].append(v)
    print('negatives: static scans + confirmed pauses + unlabelled pairs more than 2 pairs from any label')
    print('k   n_neg  median  p95   max   | thr=max_neg: moving detected (A30+A44, n=%d) | at 0.5: moving, false alarms' % len(pos[8]))
    for k in KS:
        n_, p_ = np.array(neg[k]), np.array(pos[k])
        thr = n_.max()
        print(f'{k:<3} {len(n_):<6} {np.median(n_):.2f}   {np.percentile(n_,95):.2f}  {thr:.2f}  | {np.sum(p_>thr)}/{len(p_)} ({100*np.mean(p_>thr):.0f}%)   | {np.sum(p_>=0.5)}/{len(p_)}, FA {np.sum(n_>=0.5)}/{len(n_)}')
    print('\nA44 pairs 67, 70, 71 (verified moving, missed at k=30):')
    for k in KS:
        print(f'  k={k:<3}', [round(data['A44'][f][f'k{k}'], 2) for f in (67, 70, 71)])
    print('\nLabelled sets (pauses not checked): fraction above the zero-false-alarm threshold of each k')
    for n in LABELLED:
        print(f'  {n:<6}', '  '.join(f"k{k}: {np.sum(np.array(lab[n][k])>np.max(neg[k]))}/{len(lab[n][k])}" for k in KS))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/cluster_size_2026_10')
