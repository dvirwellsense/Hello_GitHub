"""Body / background masks.

person_mask_seg  - pretrained person segmentation (YOLOv8m-seg, COCO 'person'); works for any clothing.
                   The small 's' model missed or bloated the legs on the leg scan; 'm' is ~0.6 s/frame on 4 CPU cores.
person_mask_color - HSV colour mask tuned to one volunteer (blue shirt + skin); kept for reference only.
phantom_mask     - orange chest phantom.
specular_mask    - bright, unsaturated highlights (they move with the camera, not with the surface).
"""
import cv2
import numpy as np

_models = {}


def person_mask_seg(img, weights='yolov8m-seg.pt', conf=0.25):
    from ultralytics import YOLO  # heavy import, only when used
    if weights not in _models:
        _models[weights] = YOLO(weights)
    # The patient lies along the image x axis; rotate so the person is upright for the model.
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    r = _models[weights].predict(rot, classes=[0], conf=conf, imgsz=960, retina_masks=True, verbose=False)[0]
    m = np.zeros(rot.shape[:2], np.uint8)
    if r.masks is not None and len(r.masks):
        k = int(np.argmax(r.boxes.conf.cpu().numpy()))
        m = (r.masks.data[k].cpu().numpy() > 0.5).astype(np.uint8) * 255
        m = cv2.resize(m, (rot.shape[1], rot.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _largest_component(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m)
    return np.where(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)


def person_mask_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shirt = cv2.inRange(hsv, (95, 70, 40), (125, 255, 230))
    skin = cv2.inRange(hsv, (0, 40, 70), (22, 200, 255))
    m = cv2.morphologyEx(shirt | skin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.morphologyEx(_largest_component(m), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def phantom_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 60, 80), (25, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return _largest_component(m)


def specular_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.dilate(cv2.inRange(hsv, (0, 0, 215), (180, 90, 255)), np.ones((21, 21), np.uint8))


def body_and_specular(kind, img):
    """Return (body mask, specular mask restricted to the body) for mask kind 'seg' | 'color' | 'phantom'."""
    if kind == 'phantom':
        body = phantom_mask(img)
        return body, specular_mask(img) & body
    body = person_mask_seg(img) if kind == 'seg' else person_mask_color(img)
    return body, np.zeros_like(body)
