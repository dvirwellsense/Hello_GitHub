"""Presentation plot of one evaluated scan (CSV from evaluate_scan.py).

Top: gantry angle per frame, coloured by phase (approach / stop / sweep / return); the exposure window
is shaded. Bottom: the motion score of every pair (frame b of the pair on the x axis, log scale):
  blue dot      gantry moving, NO_MOTION
  orange cross  SUSPECT (score above threshold) inside the exposure window
  hollow grey   outside the exposure window (scored, never alarms)
  grey square   gantry stopped (direct comparison)
  red tick      NO_BODY_DATA (fewer than 30 body points): no score
  yellow band   manually labelled motion (pair b frames)
Example: python scripts/plot_scan.py results/legs_A30_A44_2026_10/A30_E3u.csv --label 55-69 --label 9-24 --title "A30" --out a30.png
"""
import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PHASE_COLOR = {'approach': '#9AA7BD', 'dwell_start': '#14213D', 'sweep': '#2A78D6', 'dwell_end': '#14213D', 'return': '#9AA7BD'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--title', default='')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    f = np.array([int(r['frame']) for r in rows])
    ang = np.array([float(r['angle']) for r in rows])
    ph = np.array([r['phase'] for r in rows])
    ex = np.array([r['exposure'] == 'True' for r in rows])
    st = np.array([r['same_angle'] == 'True' for r in rows])
    ms = np.array([r.get('motion_state', '') for r in rows])
    num = lambda k: np.array([float(r[k]) if r.get(k) not in (None, '', 'nan') else np.nan for r in rows])
    score = np.where(st, num('static_cl'), num('epi_enc_cl30'))

    fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True, gridspec_kw=dict(height_ratios=(1, 2.4)))
    lo, hi = f[ex].min() - 0.5, f[ex].max() + 0.5
    for x in ax:
        x.axvspan(lo, hi, color='#E6EBF2', lw=0, zorder=0)
    for p, c in PHASE_COLOR.items():
        m = ph == p
        ax[0].plot(f[m], ang[m], 'o', ms=3, color=c)
    ax[0].set_ylabel('angle (deg)', fontsize=10)
    ax[0].text(f.min(), ang.max() * 0.85, 'grey = exposure window', fontsize=9, color='#3A4A63')
    ymin = 0.01
    for s in a.label:
        l0, l1 = map(int, s.split('-'))
        ax[1].axvspan(l0 - 0.5, l1 + 0.5, color='#EDA100', alpha=0.28, lw=0, zorder=1)
    ok = np.isfinite(score)
    sc = np.clip(score, ymin, None)
    m = ok & ~ex
    ax[1].plot(f[m], sc[m], 'o', ms=5, mfc='none', color='#9AA7BD', label='outside exposure window (never alarms)', zorder=3)
    m = ok & ex & st
    ax[1].plot(f[m], sc[m], 's', ms=5, color='#5A6880', label='gantry stopped: direct comparison', zorder=3)
    m = ok & ex & ~st & (ms != 'SUSPECT')
    ax[1].plot(f[m], sc[m], 'o', ms=5, color='#2A78D6', label='no motion', zorder=3)
    m = ok & ex & (ms == 'SUSPECT')
    ax[1].plot(f[m], sc[m], 'X', ms=10, color='#EB6834', mec='#7A2E10', mew=0.8, label='motion suspected', zorder=4)
    m = ex & (ms == 'NO_BODY_DATA')
    ax[1].plot(f[m], np.full(m.sum(), ymin * 1.3), '|', ms=14, mew=2.5, color='#C0392B', label='no body data (<30 points)', zorder=4)
    ax[1].axhline(a.threshold, color='#14213D', lw=1.2, ls='--', zorder=2)
    ax[1].text(f.min(), a.threshold * 1.15, f'threshold {a.threshold:g} px', va='bottom', fontsize=9, color='#14213D')
    ax[1].set_yscale('log')
    ax[1].set_ylim(ymin, max(20, np.nanmax(sc) * 1.5))
    ax[1].set_ylabel('motion score (px, log)', fontsize=10)
    ax[1].set_xlabel('frame (second frame of each pair)', fontsize=10)
    handles, labels = ax[1].get_legend_handles_labels()
    if a.label:
        handles.append(plt.Rectangle((0, 0), 1, 1, color='#EDA100', alpha=0.28))
        labels.append('labelled motion')
    fig.legend(handles, labels, fontsize=9, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    for x in ax:
        x.grid(axis='y', color='#E0E5EC', lw=0.8)
        for sp in ('top', 'right'):
            x.spines[sp].set_visible(False)
    if a.title:
        ax[0].set_title(a.title, fontsize=12, loc='left')
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(a.out, dpi=130)


if __name__ == '__main__':
    main()
