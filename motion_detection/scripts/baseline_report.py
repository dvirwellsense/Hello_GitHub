"""Compare runs of evaluate_scan.py --baseline L (pair i-L -> i): noise, coverage, labelled motion and false alarms.

  python scripts/baseline_report.py results/baseline_2026_10 > results/baseline_2026_10/report.txt

A pair i-L -> i spans the adjacent pairs i-L+1 .. i. It counts as moving when any of them is a labelled moving pair,
and as negative when all of them are negatives (as in compare_mask_runs.py: static scans, confirmed pauses, unlabelled
pairs more than 2 pairs from a label). k=16, threshold 0.5 px, persistence 2, as in the default settings.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_size_report as c  # noqa: E402
import compare_mask_runs as cm  # noqa: E402

THR = 0.5


def negatives(n, frames):
    near = set()
    for k in c.MOVING.get(n, set()) | c.LABELLED.get(n, set()) | c.UNVERIFIED.get(n, set()):
        near |= set(range(k - 2, k + 3))
    return {k for k in frames if n in c.STATIC or k in c.PAUSE.get(n, set()) or k not in near}


def main(root):
    root = Path(root)
    runs = sorted((int(p.name[1:]), p) for p in root.glob('L*') if p.is_dir())
    for L, d in runs:
        static, meas, tot = [], 0, 0
        mov = movn = 0
        fneg, nneg, fal = [], 0, []
        alarms = {}
        for n in cm.SCANS:
            R = cm.load(d, n)
            if R is None:
                continue
            sc = {int(r['frame']): cm.f(r['epi_enc_cl16']) for r in R}
            all_frames = set(range(min(sc) - L, max(sc) + 1))
            neg1 = negatives(n, all_frames)
            tot += len(sc)
            meas += sum(np.isfinite(v) for v in sc.values())
            if n in c.STATIC:
                static += [v for v in sc.values() if np.isfinite(v)]
            sus = {k for k, v in sc.items() if v >= THR}
            M = {k for k in sc if any(m in c.MOVING.get(n, set()) for m in range(k - L + 1, k + 1))}
            N = {k for k in sc if all(m in neg1 for m in range(k - L + 1, k + 1))}
            mov += len(sus & M)
            movn += len(M)
            nneg += len(N)
            fneg += [(n, k) for k in sorted(sus & N)]
            al = cm.alarms(sus)
            fal += [(n, k) for k in al if k in N]
            alarms[n] = al
        s = np.array(static)
        print(f'== L={L}: static scans median {np.median(s):.3f} px, 99th pct {np.percentile(s, 99):.3f}, max {s.max():.3f} '
              f'({len(s)} pairs); measurable {meas}/{tot}')
        print(f'   moving pairs flagged {mov}/{movn}; negatives flagged {len(fneg)}/{nneg} {fneg}')
        print(f'   alarm frames on negatives (persistence 2): {fal}')
        for n, al in alarms.items():
            print(f'   {n:12s} alarm frames: {al}')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/baseline_2026_10')
