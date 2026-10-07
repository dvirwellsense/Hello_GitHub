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
          'A08': R(74, 87), 'A12': R(57, 61), 'A26': R(59, 62) | R(64, 68),
          'A44': R(56, 57) | R(59, 61) | R(63, 68) | R(70, 72) | R(75, 76) | {80} | R(83, 85)}
PAUSE = {'A30': R(65, 67), 'A44': {58, 62, 69, 73, 74, 77, 78, 81, 82},
         'A08': {73, 88} | R(115, 117), 'A12': {56, 62, 63, 66}, 'A26': {63, 69}}
# labelled but pauses not checked: excluded from negatives, reported separately
LABELLED = {'A28_1': R(63, 86)}
UNVERIFIED = {'A28_1': R(106, 110)}  # suspects outside the label, not verified
HELD_OUT = ('A08', 'A12', 'A26')  # labelled after k=16 was chosen on A30/A44
STATIC = {'A07', 'A11', 'legs_static', 'VC01_bh'}


def load(d, name):
    return {int(r['frame']): {k: float(r[k]) for k in r if k != 'frame'} for r in csv.DictReader(open(Path(d) / f'{name}.csv'))}


def split(data, scans):
    """Negative / moving scores per k for the given scans."""
    neg, pos = {k: [] for k in KS}, {k: [] for k in KS}
    for n in scans:
        marked = MOVING.get(n, set()) | LABELLED.get(n, set()) | UNVERIFIED.get(n, set())
        near = set()
        for f in marked:
            near |= {f - 2, f - 1, f, f + 1, f + 2}
        for f, row in data[n].items():
            for k in KS:
                v = row[f'k{k}']
                if np.isnan(v):
                    continue
                if n in STATIC or f in PAUSE.get(n, set()) or f not in near:
                    neg[k].append(v)
                elif f in MOVING.get(n, set()):
                    pos[k].append(v)
    return neg, pos


def table(title, neg, pos):
    print(title)
    print('k   n_neg  median  p95   max   | moving detected at 0.5 | false alarms at 0.5 | moving above max negative')
    for k in KS:
        n_, p_ = np.array(neg[k]), np.array(pos[k])
        print(f'{k:<3} {len(n_):<6} {np.median(n_):.2f}   {np.percentile(n_, 95):.2f}  {n_.max():.2f}  | {np.sum(p_ >= 0.5)}/{len(p_)}'
              f'{"":<14}| {np.sum(n_ >= 0.5)}/{len(n_)}{"":<12}| {np.sum(p_ > n_.max())}/{len(p_)}')


def main(d):
    names = ['A30', 'A44', 'A08', 'A12', 'A26', 'A28_1', 'A07', 'A11', 'legs_static', 'VC01_bh']
    data = {n: load(d, n) for n in names}
    neg, pos = split(data, names)
    table('ALL scans. negatives: static scans + confirmed pauses + unlabelled pairs more than 2 pairs from any label', neg, pos)
    others = [n for n in names if n not in HELD_OUT]
    table('\nDEVELOPMENT set (k=16 was chosen here): all scans except A08, A12, A26', *split(data, others))
    held = [n for n in names if n in HELD_OUT]
    table('\nHELD-OUT: A08, A12, A26 (labelled after k=16 was chosen)', *split(data, held))
    print('\nA44 pairs 67, 70, 71 (confirmed moving, missed at k=30):')
    for k in KS:
        print(f'  k={k:<3}', [round(data['A44'][f][f'k{k}'], 2) for f in (67, 70, 71)])
    print('\nA26 (9 moving pairs: 59-62, 64-68; quiet: 63, 69), score per pair at k=16 and k=30:')
    for k in (16, 30):
        print(f'  k={k}:', {f: round(data['A26'][f][f'k{k}'], 2) for f in range(58, 70)})
    negs = split(data, names)[0]
    print('\nLabelled but pauses not checked: A28_1 63-86, pairs above the largest negative of each k')
    for k in KS:
        v = np.array([data['A28_1'][f][f'k{k}'] for f in LABELLED['A28_1'] if not np.isnan(data['A28_1'][f][f'k{k}'])])
        print(f'  k={k:<3} {np.sum(v > max(negs[k]))}/{len(v)}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/cluster_size_2026_10')
