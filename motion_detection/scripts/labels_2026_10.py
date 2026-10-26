"""Labels of October 2026, one place: what the labeller stated per scan, and the label of every pair.

A pair is named by its target frame b (pair a -> b, a = b - 1). Label of a pair and its type:
  motion               exact          labeller confirmed motion in this pair (cluster_size_report.MOVING)
  no_motion            exact          labeller confirmed no motion in this pair (PAUSE, and A44 90)
  motion_range         general_range  inside a stated motion range whose pauses were not checked (A28_1 63-86)
  motion_outside_window general_range stated motion before the arc reached the start angle (A30 9-24, legs_static 3-8)
  no_motion_scan       general_range  static control scan: "no motion" stated for the whole scan, not pair by pair
  suspect_unverified   unknown        suspects of an earlier run that the labeller did not check (A28_1 106-110)
  unlabelled           unknown        nothing stated about this pair
Scan A16 was never received or analysed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_size_report as c  # noqa: E402

R = lambda a, b: set(range(a, b + 1))

SCANS = {
    'A30': dict(folder='arc22_office_20260907_151756', what='Volunteer, right leg moves',
                statement='Motion from transition 54->55 to 68->69 (pairs 55-69); an earlier motion from 8->9 to 23->24, before the arc '
                          'reached the start angle (outside the exposure window). Later confirmed: 65-67 is a pause.'),
    'A44': dict(folder='arc22_office_20260907_152109', what='Volunteer, left knee raised and moved sideways',
                statement='Motion 55->56 to 71->72, 74->76, 79->85. Confirmed pauses 57->58, 61->62, 68->69, 72->73, 73->74, 76->77, '
                          '77->78, 80->81, 81->82. Pair 67 is not a pause (blur at the knee). No motion 89->90.'),
    'A08': dict(folder='arc22_office_20260907_125020', what='Volunteer, legs, right knee moves to the right',
                statement='Motion from 73->74 (small) to 86->87. No motion 72->73, 87->88, around 116.'),
    'A12': dict(folder='arc22_office_20260907_125615', what='Volunteer, knee, small rotational motion',
                statement='Motion from 56->57 to 60->61. No motion 65->66; 55->56, 61->62, 62->63 confirmed quiet.'),
    'A26': dict(folder='arc22_office_20260907_142049', what='Small hand motion (3-5 mm)',
                statement='Motion from 58->59 to 67->68; no motion 62->63 and 68->69, small motion in the other pairs of the range.'),
    'A28_1': dict(folder='arc22_office_20260907_151502', what='Chest, large motion',
                  statement='Significant motion in frames 63-86 (pairs 63-86); pauses inside not checked. Suspects 106-110 not verified.'),
    'A07': dict(folder='arc22_office_20260907_124742', what='Volunteer, legs, no motion', statement='No motion.'),
    'A11': dict(folder='arc22_office_20260907_125440', what='Volunteer, knee, no motion', statement='No motion.'),
    'legs_static': dict(folder='arc22_office_20260830_142447_volunteer_both_legs_manual_normal_static_repea', what='Volunteer, both legs, static',
                        statement='No motion during the scan; the legs moved at HOME before the arc started (pairs 3-8, outside the window).'),
    'VC01_bh': dict(folder='arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post', what='Volunteer, chest, breath hold',
                    statement='Breath hold; no deliberate motion.'),
    'A27': dict(folder='arc22_office_20260907_150841', what='Volunteer V03, chest, breath hold',
                statement='SCAN_LABEL: breath hold maintained throughout the active sweep; expected NO MOTION.'),
}
NO_MOTION = {n: set(v) for n, v in c.PAUSE.items()}
NO_MOTION['A44'] = NO_MOTION['A44'] | {90}
OUTSIDE_WINDOW_MOTION = {'A30': R(9, 24), 'legs_static': R(3, 8)}
STATIC = set(c.STATIC) | {'A27'}


def pair_label(scan, b):
    """(label, type) of pair b-1 -> b."""
    if b in c.MOVING.get(scan, set()):
        return 'motion', 'exact'
    if b in NO_MOTION.get(scan, set()):
        return 'no_motion', 'exact'
    if b in c.LABELLED.get(scan, set()):
        return 'motion_range', 'general_range'
    if b in OUTSIDE_WINDOW_MOTION.get(scan, set()):
        return 'motion_outside_window', 'general_range'
    if b in c.UNVERIFIED.get(scan, set()):
        return 'suspect_unverified', 'unknown'
    if scan in STATIC:
        return 'no_motion_scan', 'general_range'
    return 'unlabelled', 'unknown'


def negative(scan, b):
    """Negative in the counts of compare_mask_runs.py: static scan, confirmed pause, or unlabelled and more than
    2 pairs away from any motion label."""
    if scan in STATIC or b in c.PAUSE.get(scan, set()):
        return True
    marked = c.MOVING.get(scan, set()) | c.LABELLED.get(scan, set()) | c.UNVERIFIED.get(scan, set())
    return all(abs(b - m) > 2 for m in marked)
