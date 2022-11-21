import numpy as np
import torch


def horizontal_expand_rightward(label, feature, to_expand=20):
    if isinstance(label, torch.Tensor):
        label_copy = label.clone()
    else:
        label_copy = label.copy()
    x, y = np.where(label_copy == feature)
    xacc, yacc = x, y
    for i in range(to_expand):
        xacc = np.concatenate([xacc, x])
        yacc = np.concatenate([yacc, y+i])
    label_copy[xacc, yacc] = feature
    return label_copy


def vertical_expand_upward(label, feature):
    """Fill all space above the appeared feature
    with feature

    label: expected to be of shape (height, width)
    """
    if isinstance(label, torch.Tensor):
        label_copy = label.clone()
    else:
        label_copy = label.copy()
    height, width = label.shape
    for w in range(width):
        feature_h = np.where(label[:, w] == feature)[0]
        if len(feature_h) == 0:
            continue
        max_h = feature_h[-1]
        label_copy[:max_h, w] = feature
    return label_copy


def expand_label(label, instrument_label=2, mirror_label=4, expansion_instrument=30,
                 expansion_mirror=60, expand_upward=False):
    """Expand the line-shaped instrument label and shadow label into greater but
    inaccurate segmentation mask that covers the whole instrument and shadow.

    Args:
    label: input size is expected to be [h, w]
    """
    # For label 2 4 (instrument & its mirroring), we horizontally expand
    # a couple of pixels rightward
    label = horizontal_expand_rightward(label, instrument_label,
                                        to_expand=expansion_instrument)
    # shadows are generally broader
    label = horizontal_expand_rightward(
        label, mirror_label, to_expand=expansion_mirror)
    if expand_upward:
        label = vertical_expand_upward(label, mirror_label)
        label = vertical_expand_upward(label, instrument_label)
    return label


def get_dst_shadow(src_label, dst_label, instrument_label=2, mirror_label=4,
                   top_layer_label=1,  img_width=256, img_height=256):
    """Get the shadow of the source label in the destination label, taking
    the instrument and shadow label as well as the layer label in the destination
    into account
    """
    shadow_x = np.array([], dtype=np.int64)
    shadow_y = np.array([], dtype=np.int64)
    # Requirements for the shadow label:
    # 1. Horizontally after the starting of the instrument/mirroring & before the
    #    ending of the instrument/mirroring
    # 2. Vertically below the upper bound of layers
    x, y = np.where(np.logical_or(src_label == instrument_label,
                    src_label == mirror_label))  # (1024, 512)
    if len(x) == 0:
        return shadow_x, shadow_y
    left_bound = np.min(y)
    right_bound = np.max(y)
    accumulated_min_lowerbound = 0
    for i in range(left_bound, right_bound):
        instrument_above = np.where(dst_label[:, i] == top_layer_label)[0]
        if len(instrument_above) == 0:
            if accumulated_min_lowerbound == 0:
                continue
            else:
                # set to current recorded lowest shadow
                instrument_lowerbound = accumulated_min_lowerbound
        else:
            # print("instrument_above", instrument_above, len(instrument_above))
            instrument_lowerbound = np.max(instrument_above)
            if accumulated_min_lowerbound == 0:
                # initialize
                accumulated_min_lowerbound = instrument_lowerbound
            else:
                accumulated_min_lowerbound = max(
                    accumulated_min_lowerbound, instrument_lowerbound)
        x_vertical = np.arange(instrument_lowerbound,
                               img_height)  # upperbound to bottom
        y_vertical = np.full_like(x_vertical, i)
        shadow_x = np.concatenate([shadow_x, x_vertical])
        shadow_y = np.concatenate([shadow_y, y_vertical])
    return shadow_x, shadow_y
