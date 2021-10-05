# ====================================
# IMPORT MODULES
# ====================================

import tensorflow as tf


# ====================================
# FUNCTIONS DEFINITION
# ====================================

def iou(box_1, box_2):
    """
    Calculate intersection over union
    Box value: (x1, y1, x2, y2)
    :param box_1: Bounding box predictions (batch_size, 4)
    :param box_2: Ground truth boxes (batch_size, 4)
    :return : IOU over all examples
    """

    # Find the intersection coordinate
    x1 = tf.maximum(box_1[..., 0], box_2[..., 0])
    y1 = tf.maximum(box_1[..., 1], box_2[..., 1])
    x2 = tf.minimum(box_1[..., 2], box_2[..., 2])
    y2 = tf.minimum(box_1[..., 3], box_2[..., 3])

    # Compute area
    inter_area = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    box_1_area = tf.abs((box_1[..., 2] - box_1[..., 0]) *
                        (box_1[..., 3] - box_1[..., 1]))
    box_2_area = tf.abs((box_2[..., 2] - box_1[..., 0]) *
                        (box_2[..., 3] - box_1[..., 1]))

    return inter_area / (box_1_area + box_2_area - inter_area)
