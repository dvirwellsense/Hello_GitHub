"""Tables of the October 2026 release from the run folders (no image processing).

  python scripts/release_tables.py --runs <runs_dir> --out <tables_dir>

<runs_dir> holds the folders written by the release runs (see the release README):
  regular/<scan>.csv          evaluate_scan.py --chain-L 0 (adjacent pairs only, persistence 2)
  combined/<scan>.csv         evaluate_scan.py defaults (adjacent pairs OR chained L=4)
  mediapipe_selfie/, mediapipe_multiclass/, yolov8n640_every4/   the same defaults with other saved masks
  realtime_*/<scan>.csv       scripts/realtime_run.py
Writes per_pair_results.csv, comparison_*.csv/.md, masks_per_scan.csv/.md, realtime_timing.csv/.md.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import labels_2026_10 as L

SCANS = ['A30', 'A44', 'A08', 'A12', 'A26', 'A28_1', 'A07', 'A11', 'legs_static', 'VC01_bh', 'A27']
THR, CTHR, CL = 0.5, 1.2, 4


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def tb(v):
    return v == 'True' or v is True


def load(d, n):
    p = Path(d) / f'{n}.csv'
    return list(csv.DictReader(open(p))) if p.exists() else None


def fmt(v, nd=4):
    return '' if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(v, nd) if isinstance(v, float) else v)


def rng(v):
    v = sorted(v)
    out, s, p = [], None, None
    for x in v:
        if s is None:
            s = p = x
        elif x == p + 1:
            p = x
        else:
            out.append((s, p))
            s = p = x
    if s is not None:
        out.append((s, p))
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in out) or '-'


def per_pair(n, comb, reg):
    """One row per pair a -> b of the combined run, with the regular run's alarm."""
    regalarm = {int(r['frame']): tb(r['alarm']) for r in reg}
    frames = [int(r['frame']) for r in comb]
    out = []
    for k, r in enumerate(comb):
        b = frames[k]
        a = frames[k - 1] if k else b - 1
        lab, typ = L.pair_label(n, b)
        same = tb(r['same_angle'])
        win = tb(r['exposure'])
        score = fl(r['static_cl'] if same else r['epi_enc_cl16'])
        nb = int(fl(r['n_body']))
        if np.isfinite(score):
            reason = ''
        elif same:
            reason = 'same-angle pair: fewer than 30 body points for the direct comparison'
        else:
            reason = f'fewer than {8 if nb < 8 else 16} body points above the detector (k=16 cluster)'
        csrc = frames[k - CL] if k >= CL else None
        cs = fl(r.get('chain_cl16'))
        nc = int(fl(r.get('n_chain'))) if np.isfinite(fl(r.get('n_chain'))) else 0
        span = comb[k - CL + 1:k + 1] if k >= CL - 1 else []
        if k < CL:
            creason = 'no frame i-4 (start of the scan)'
        elif not np.isfinite(cs):
            creason = 'fewer than 16 points survive the chained tracking'
        else:
            creason = ''
        if k < CL - 1:
            celig = 'start of the scan'
        elif not all(tb(x['exposure']) for x in span):
            celig = 'chain leaves the exposure window'
        elif any(tb(x['same_angle']) for x in span):
            celig = 'chain contains a same-angle pair'
        else:
            celig = ''
        out.append(dict(
            scan=n, frame_a=a, frame_b=b, in_exposure_window=int(win), phase=r['phase'], same_angle=int(same),
            angle_b_deg=fmt(fl(r['angle']), 3), label=lab, label_type=typ, negative_in_counts=int(win and L.negative(n, b)),
            adj_score_kind='direct (same angle, displacement)' if same else 'epipolar k=16', adj_score_px=fmt(score),
            adj_n_points=nb, adj_measurable=int(np.isfinite(score)), adj_not_measurable_reason=reason,
            adj_suspect=int(r['motion_state'] == 'SUSPECT'), adj_suspect_in_window=int(r['motion_state'] == 'SUSPECT' and win),
            alarm_regular=int(regalarm.get(b, False)),
            chain_frame_a=csrc if csrc is not None else '', chain_score_px=fmt(cs), chain_n_points=nc,
            chain_measurable=int(np.isfinite(cs)), chain_not_measurable_reason=creason,
            chain_eligible=int(celig == ''), chain_not_eligible_reason=celig,
            chain_above_threshold=int(np.isfinite(cs) and cs > CTHR), chain_suspect=int(tb(r['chain_suspect'])),
            alarm_chain=int(tb(r['alarm_chain'])), alarm_final=int(tb(r['alarm'])),
            check_alarm_adjacent_equals_regular=int(tb(r['alarm_adjacent']) == regalarm.get(b, False))))
    return out


def segments(pairs):
    s = sorted(pairs)
    out, cur = [], []
    for x in s:
        if cur and x == cur[-1] + 1:
            cur.append(x)
        else:
            if cur:
                out.append(cur)
            cur = [x]
    if cur:
        out.append(cur)
    return out


EPISODES = {'A30': [(55, 69)], 'A44': [(56, 72), (75, 76), (80, 85)], 'A08': [(74, 87)], 'A12': [(57, 61)],
            'A26': [(59, 68)], 'A28_1': [(63, 86)]}


def comparison(rows):
    """Regular vs combined on the same pairs, masks and labels (exposure window only)."""
    by = {}
    for r in rows:
        by.setdefault(r['scan'], []).append(r)
    summary, events, extra = [], [], []
    for n in SCANS:
        R = [r for r in by[n] if r['in_exposure_window']]
        idx = {r['frame_b']: r for r in by[n]}
        mov = [r for r in R if r['label'] == 'motion']
        neg = [r for r in R if r['negative_in_counts']]

        def spans_motion(r):
            return any(idx.get(x, {}).get('label') in ('motion', 'motion_range') for x in range(r['frame_b'] - CL + 1, r['frame_b'] + 1))
        summary.append(dict(
            scan=n, window_pairs=len(R), adj_measurable=sum(r['adj_measurable'] for r in R),
            chain_measurable_and_eligible=sum(r['chain_measurable'] and r['chain_eligible'] for r in R),
            motion_pairs_exact=len(mov), motion_suspect_regular=sum(r['adj_suspect'] for r in mov),
            motion_alarm_regular=sum(r['alarm_regular'] for r in mov), motion_alarm_combined=sum(r['alarm_final'] for r in mov),
            negatives=len(neg), negative_suspect_regular=sum(r['adj_suspect'] for r in neg),
            negative_alarm_regular=sum(r['alarm_regular'] for r in neg), negative_chain_suspect=sum(r['chain_suspect'] for r in neg),
            negative_alarm_combined=sum(r['alarm_final'] for r in neg),
            negative_alarm_added_chain_spans_motion=sum(r['alarm_final'] and not r['alarm_regular'] and spans_motion(r) for r in neg),
            negative_alarm_added_no_motion_in_chain=sum(r['alarm_final'] and not r['alarm_regular'] and not spans_motion(r) for r in neg),
            alarm_frames_regular=rng([r['frame_b'] for r in R if r['alarm_regular']]),
            alarm_frames_added_by_chain=rng([r['frame_b'] for r in R if r['alarm_final'] and not r['alarm_regular']])))
        # events: stated episodes and, inside them, segments of consecutive exact motion pairs
        for lo, hi in EPISODES.get(n, []):
            ep = [idx[b] for b in range(lo, hi + 1) if b in idx]
            kind = 'general_range' if n == 'A28_1' else 'exact'
            segs = [s for s in segments([r['frame_b'] for r in ep if r['label'] in ('motion', 'motion_range')])]
            for s in ([list(range(lo, hi + 1))] + (segs if len(segs) > 1 else [])):
                rr = [idx[b] for b in s if b in idx]
                fr = [r['frame_b'] for r in rr if r['alarm_regular']]
                fc = [r['frame_b'] for r in rr if r['alarm_final']]
                first_c = min(fc) if fc else None
                note = ''
                if not fr and fc:
                    span = list(range(first_c - CL + 1, first_c + 1))
                    before = [b for b in span if b < s[0] and idx.get(b, {}).get('label') in ('motion', 'motion_range')]
                    if before:
                        note = (f'chain i-4 -> i at {first_c} also spans earlier motion pairs {rng(before)}: '
                                'cannot be attributed to this segment alone')
                events.append(dict(scan=n, level='episode' if s == list(range(lo, hi + 1)) else 'segment', pairs=f'{s[0]}-{s[-1]}',
                                   label_type=kind, detected_regular=int(bool(fr)), first_alarm_regular=min(fr) if fr else '',
                                   detected_combined=int(bool(fc)), first_alarm_combined=first_c if fc else '',
                                   newly_detected=int(bool(fc) and not fr), note=note))
        # every alarm frame added by the chain, classified
        for r in R:
            if not (r['alarm_final'] and not r['alarm_regular']):
                continue
            b = r['frame_b']
            span = [idx[x] for x in range(b - CL + 1, b + 1) if x in idx]
            mot = [x['frame_b'] for x in span if x['label'] in ('motion', 'motion_range')]
            unv = [x['frame_b'] for x in span if x['label'] == 'suspect_unverified']
            if r['label'] == 'motion':
                cls = 'motion pair without a regular alarm'
            elif r['label'] == 'motion_range':
                cls = 'inside a general motion range, without a regular alarm'
            elif mot:
                cls = 'alarm continues after / between motion (the chain still spans labelled motion)'
            elif unv:
                cls = 'chain spans unverified suspects'
            else:
                cls = 'no labelled motion in the chain'
            extra.append(dict(scan=n, frame_b=b, chain_frame_a=r['chain_frame_a'], pair_label=r['label'], label_type=r['label_type'],
                              chain_score_px=r['chain_score_px'], adj_score_px=r['adj_score_px'], adj_suspect=r['adj_suspect'], class_=cls,
                              motion_pairs_in_chain=rng(mot), unverified_in_chain=rng(unv)))
    return summary, events, extra


def masks_table(runs, ref_rows):
    """Per scan and mask source: measurable pairs, missing pairs, motion / negatives, alarms vs YOLOv8m union."""
    out = []
    ref = {}
    for r in ref_rows:
        ref.setdefault(r['scan'], {})[r['frame_b']] = r
    for tag, d in runs.items():
        for n in SCANS:
            R = load(d, n)
            if R is None:
                continue
            rows = per_pair(n, R, R)  # regular alarm column = this run's adjacent alarm
            W = [r for r in rows if r['in_exposure_window']]
            miss = [r['frame_b'] for r in W if not r['adj_measurable']]
            miss_ref = [b for b, r in ref[n].items() if r['in_exposure_window'] and not r['adj_measurable']]
            mov = [r for r in W if r['label'] == 'motion']
            neg = [r for r in W if r['negative_in_counts']]
            al = {r['frame_b'] for r in W if r['alarm_final']}
            al_ref = {b for b, r in ref[n].items() if r['in_exposure_window'] and r['alarm_final']}
            out.append(dict(mask=tag, scan=n, window_pairs=len(W), measurable=len(W) - len(miss),
                            missing_pairs=rng(miss), missing_pairs_yolov8m=rng(miss_ref),
                            motion_pairs=len(mov), motion_suspect=sum(r['adj_suspect'] for r in mov),
                            motion_missed=rng([r['frame_b'] for r in mov if not r['adj_suspect']]),
                            motion_alarm_final=sum(r['alarm_final'] for r in mov),
                            negatives=len(neg), negative_suspect=sum(r['adj_suspect'] for r in neg),
                            negative_alarm_final=sum(r['alarm_final'] for r in neg),
                            alarm_frames_lost_vs_yolov8m=rng(al_ref - al), alarm_frames_new_vs_yolov8m=rng(al - al_ref)))
    return out


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return dict(n=0, mean='', p95='', max='')
    return dict(n=len(v), mean=round(float(v.mean()), 1), p95=round(float(np.percentile(v, 95)), 1), max=round(float(v.max()), 1))


def realtime_table(runs, warmup=3):
    out = []
    for tag, d in runs.items():
        allrows = []
        for n in SCANS:
            R = load(d, n)
            if R is None:
                continue
            allrows += [(n, k, r) for k, r in enumerate(R)]
        for scope in SCANS + ['ALL']:
            rows = [(k, r) for n, k, r in allrows if scope in ('ALL', n)]
            if not rows:
                continue
            steady = [r for k, r in rows if k >= warmup]
            wu = [r for k, r in rows if k < warmup]
            mk = [r for r in steady if r['mask_computed'] == '1']
            ru = [r for r in steady if r['mask_computed'] != '1']
            row = dict(config=tag, scan=scope, frames=len(rows), warmup_frames=len(wu),
                       warmup_total_max_ms=round(max(fl(r['t_total_ms']) for r in wu), 1) if wu else '')
            for name, sel, key in (('total', steady, 't_total_ms'), ('total_mask_frames', mk, 't_total_ms'),
                                   ('total_reuse_frames', ru, 't_total_ms'), ('decode', steady, 't_decode_ms'),
                                   ('mask', mk, 't_mask_ms'), ('monitor', steady, 't_monitor_ms')):
                s = stats([fl(r[key]) for r in sel])
                row.update({f'{name}_n': s['n'], f'{name}_mean_ms': s['mean'], f'{name}_p95_ms': s['p95'], f'{name}_max_ms': s['max']})
            row['frames_over_250ms'] = sum(fl(r['t_total_ms']) > 250 for r in steady)
            out.append(row)
    return out


def write(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)


def md(rows, cols, path, title=''):
    lines = ([f'## {title}', ''] if title else []) + ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in SCANS:
        rows += per_pair(n, load(runs / 'combined', n), load(runs / 'regular', n))
    bad = [(r['scan'], r['frame_b']) for r in rows if not r['check_alarm_adjacent_equals_regular']]
    print('regular run alarm == adjacent alarm of the combined run:', 'yes' if not bad else f'NO {bad[:10]}')
    write(rows, out / 'per_pair_results.csv')
    summary, events, extra = comparison(rows)
    write(summary, out / 'comparison_per_scan.csv')
    write(events, out / 'comparison_events.csv')
    write(extra, out / 'comparison_alarms_added_by_chain.csv')
    md(summary, list(summary[0]), out / 'comparison_per_scan.md')
    md(events, list(events[0]), out / 'comparison_events.md')
    md(extra, list(extra[0]), out / 'comparison_alarms_added_by_chain.md')
    mruns = {k: runs / k for k in ('combined', 'mediapipe_selfie', 'mediapipe_multiclass', 'yolov8n640_every4') if (runs / k).exists()}
    mt = masks_table(mruns, rows)
    write(mt, out / 'masks_per_scan.csv')
    md(mt, list(mt[0]), out / 'masks_per_scan.md')
    rt = {p.name: p for p in sorted(runs.glob('realtime_*')) if p.is_dir()}
    if rt:
        tt = realtime_table(rt)
        write(tt, out / 'realtime_timing.csv')
        cols = ['config', 'scan', 'frames', 'total_mean_ms', 'total_p95_ms', 'total_max_ms', 'total_mask_frames_n', 'total_mask_frames_mean_ms',
                'total_mask_frames_p95_ms', 'total_mask_frames_max_ms', 'total_reuse_frames_mean_ms', 'total_reuse_frames_p95_ms',
                'total_reuse_frames_max_ms', 'decode_mean_ms', 'mask_mean_ms', 'monitor_mean_ms', 'monitor_p95_ms', 'monitor_max_ms',
                'frames_over_250ms', 'warmup_frames', 'warmup_total_max_ms']
        md([r for r in tt if r['scan'] == 'ALL'], cols, out / 'realtime_timing_all.md')
        md(tt, cols, out / 'realtime_timing_per_scan.md')
    print('tables in', out)


if __name__ == '__main__':
    main()
