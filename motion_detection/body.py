import numpy as np, cv2
def person_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shirt = cv2.inRange(hsv, (95, 70, 40), (125, 255, 230))
    skin = cv2.inRange(hsv, (0, 40, 70), (22, 200, 255))
    m = cv2.morphologyEx(shirt | skin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.where(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
