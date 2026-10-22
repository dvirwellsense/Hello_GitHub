"""Compare evaluation runs made with different body masks (results/<run>/<scan>.csv, k=16): confirmed moving pairs flagged, flagged
negatives (static scans, confirmed pauses, unlabelled pairs more than 2 pairs from a label), measurable pairs, and the alarm frames with
persistence 2. Labels come from scripts/cluster_size_report.py.

  python scripts/compare_mask_runs.py results/k16_2026_10 results/yolov8n_2026_10 results/selfie_mediapipe_2026_10
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_size_report as c  # noqa: E402

SCANS = ['A30', 'A44', 'A08', 'A12', 'A26', 'A28_1', 'A07', 'A11', 'legs_static', 'VC01_bh', 'A27']


def load(d, n):
    p = Path(d) / f'{n}.csv'
    if not p.exists():
        return None
    return [r for r in csv.DictReader(open(p)) if r['exposure'] == 'True']


def f(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def alarms(sus, p=2):
    out, cur = [], []
    for k in sorted(sus):
        if cur and k == cur[-1] + 1:
            cur.append(k)
        else:
            if cur:
                out.append(cur)
            cur = [k]
    if cur:
        out.append(cur)
    return [k for r in out if len(r) >= p for k in r[p - 1:]]


def main(dirs):
    for d in dirs:
        mov = movn = neg = negn = meas = tot = 0
        fa = []
        al = {}
        missing = []
        for n in SCANS:
            R = load(d, n)
            if R is None:
                missing.append(n)
                continue
            sus = {int(r['frame']) for r in R if f(r['epi_enc_cl16']) >= 0.5}
            tot += len(R)
            meas += sum(1 for r in R if f(r['epi_enc_cl16']) == f(r['epi_enc_cl16']))
            M = c.MOVING.get(n, set())
            mov += len(sus & M)
            movn += len(M)
            marked = M | c.LABELLED.get(n, set()) | c.UNVERIFIED.get(n, set())
            near = set()
            for k in marked:
                near |= {k - 2, k - 1, k, k + 1, k + 2}
            for r in R:
                k = int(r['frame'])
                if n in c.STATIC or k in c.PAUSE.get(n, set()) or k not in near:
                    negn += 1
                    if k in sus:
                        fa.append((n, k))
            al[n] = alarms(sus)
        print(f'== {d}: moving flagged {mov}/{movn}; negatives flagged {len(fa)}/{negn} {fa}; measurable pairs {meas}/{tot}' + (f'; missing scans {missing}' if missing else ''))
        for n in SCANS:
            if n in al:
                print(f'   {n:12s} alarm frames (persistence 2): {al[n]}')


if __name__ == '__main__':
    main(sys.argv[1:])
