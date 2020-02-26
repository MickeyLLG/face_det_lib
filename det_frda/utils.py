import numpy as np
import cv2


def plot_vertices(image, vertices):
    image = image.copy()
    vertices = np.round(vertices).astype(np.int32)
    ls = [1446, 3626, 5178, 5965, 4803, 3257, 10083, 10705, 12121, 14587, 12915, 11753]
    # anchors = [(160, 354), (176, 345), (205, 345), (223, 364), (204, 365), (177, 363),
    #            (298, 366), (317, 346), (346, 346), (363, 359), (349, 367), (325, 370)]

    for i in ls:
        st = vertices[:2, i]
        image = cv2.circle(image, (st[0], st[1]), 0, (128, 128, 0), 4)
    return image


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    ret_roi = [sx, sy, ex, ey]
    return res, ret_roi
