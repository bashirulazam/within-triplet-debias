import math
from math import floor
import numpy as np

def encode_box(bbox_gqa, org_h, org_w, im_long_size,image_id):
    #bbox_gqa has xmin, ymin, width, height format
    x = bbox_gqa[0]
    y = bbox_gqa[1]
    w = bbox_gqa[2]
    h = bbox_gqa[3]

    if w == 0:
        w = 1
    if h == 0:
        h = 1

    scale = float(im_long_size) / max(org_h, org_w)
    image_size = im_long_size
    # recall: x,y are 1-indexed
    x, y = math.floor(scale*(x)), math.floor(scale*(y))
    w, h = math.ceil(scale*w), math.ceil(scale*h)

    # clamp to image
    if x < 0: x = 0
    if y < 0: y = 0

    # box should be at least 2 by 2
    if x > image_size - 3:
        x = image_size - 3
    if y > image_size - 3:
        y = image_size - 3
    if x + w >= image_size:
        w = image_size - x
    if y + h >= image_size:
        h = image_size - y

    xcenter = x + floor(w/2)
    ycenter = y + floor(h/2)

    # also convert to centercoordinated format
    box = np.asarray([xcenter, ycenter, w, h], dtype=np.int32)
    if box[2] <= 0:
        print(image_id)
    assert box[2] > 0  # width height should be positive numbers
    if box[3] <=0:
        print(image_id)
    assert box[3] > 0
    return box

