# ===================================================
# IMPORT MODULES
# ===================================================

import tensorflow as tf
from tensorflow.keras.losses import Loss
from utils import iou, convert_cellbox_to_corner_bbox

# ===================================================
#  CONSTANT DEFINITION
# ===================================================

coord_weight, noobj_weight = 5, 0.5


# ===================================================
#  METHOD DEFINITION
# ===================================================


def compute_xy_loss(target_xy, box1_xy, box2_xy, mask, best_iou):
    """
    Compute xy loss
    :param target_xy: Target xy offset
    :param box1_xy: Prediction xy offset from box 1
    :param box2_xy: Prediction xy from box 2
    :param mask: Tensor to determines which grid cell contains obj.
    :param best_iou: Tensor to determines which bounding box is a predictor
    :return xy_loss: xy loss
    """

    sse_xy_1 = tf.reduce_sum(tf.square(target_xy - box1_xy), -1)
    sse_xy_2 = tf.reduce_sum(tf.square(target_xy - box2_xy), -1)

    xy_predictor_1 = sse_xy_1 * mask * (1 - best_iou)
    xy_predictor_2 = sse_xy_2 * mask * best_iou
    xy_predictor = xy_predictor_1 + xy_predictor_2

    xy_loss = tf.reduce_mean(tf.reduce_sum(xy_predictor, [1, 2]))

    return xy_loss


def compute_wh_loss(target_wh, box1_wh, box2_wh, mask, best_iou):
    """
    Compute wh loss
    :param target_wh: Target xy offset
    :param box1_wh: wh offset prediction from box 1
    :param box2_wh: wh offset prediction from box 2
    :param mask: Tensor to determines which grid cell contains obj
    :param best_iou: Tensor to determines which bounding box is a predictor
    :return wh_loss: wh loss
    """

    target_wh = tf.sqrt(target_wh)
    box1_wh, box2_wh = tf.sqrt(tf.abs(box1_wh)), tf.sqrt(tf.abs(box2_wh))

    sse_wh_1 = tf.reduce_sum(tf.square(target_wh - box1_wh), -1)
    sse_wh_2 = tf.reduce_sum(tf.square(target_wh - box2_wh), -1)

    wh_predictor_1 = sse_wh_1 * mask * (1 - best_iou)
    wh_predictor_2 = sse_wh_2 * mask * best_iou
    wh_predictor = wh_predictor_1 + wh_predictor_2

    wh_loss = tf.reduce_mean(tf.reduce_sum(wh_predictor, [1, 2]))

    return wh_loss


def compute_obj_loss(target_obj, box1_obj, box2_obj, best_iou):
    """
    Compute obj. loss
    :param target_obj: Target obj (1 if cell contains obj. otherwise 0)
    :param box1_obj: Obj. prediction from box 1
    :param box2_obj: Obj. prediction from box 2
    :param best_iou: Tensor to determines which bounding box is a predictor
    :return obj_loss: obj loss
    """

    pred_obj_1 = box1_obj * target_obj * (1 - best_iou)
    pred_obj_2 = box2_obj * target_obj * best_iou
    pred_obj = pred_obj_1 + pred_obj_2

    sqrt_err_obj = tf.square(target_obj - pred_obj)

    obj_loss = tf.reduce_mean(tf.reduce_sum(sqrt_err_obj, [1, 2]))

    return obj_loss


def compute_no_obj_loss(target_obj, box1_obj, box2_obj):
    """
    Compute no obj. loss
    :param target_obj: Target obj (1 if cell contains obj. otherwise 0)
    :param box1_obj: Obj. prediction from box 1
    :param box2_obj: Obj. prediction from box 2
    :return no_obj_loss: no obj loss
    """

    target_no_obj_mask = 1 - target_obj

    pred_no_obj_1 = box1_obj * target_no_obj_mask
    pred_no_obj_2 = box2_obj * target_no_obj_mask

    sqr_err_no_obj_1 = tf.square((target_obj * target_no_obj_mask) - pred_no_obj_1)
    sqr_err_no_obj_2 = tf.square((target_obj * target_no_obj_mask) - pred_no_obj_2)
    sqr_err_no_obj = sqr_err_no_obj_1 + sqr_err_no_obj_2

    no_obj_loss = tf.reduce_mean(tf.reduce_sum(sqr_err_no_obj, [1, 2]))

    return no_obj_loss


def compute_class_dist_loss(target_cls, pred_cls, mask):
    """
    Compute class distribution loss
    :param target_cls: Target class distribution
    :param pred_cls: Class prediction
    :param mask: Tensor to determines which cell has obj
    :return cls_loss: Class distribution loss
    """

    sse_cls = tf.reduce_sum(tf.square(target_cls - pred_cls), -1)
    sse_cls = sse_cls * mask

    cls_loss = tf.reduce_mean(tf.reduce_sum(sse_cls, [1, 2]))

    return cls_loss


# ===================================================
# IMPORT MODULES
# ===================================================

class YoloLoss(Loss):
    """YOLO v1 loss"""

    def call(self, y_true, y_pred):
        """
        Compute yolo loss
        :param y_true: y target
        :param y_pred: y predict
        :return loss: YOLO loss
        """

        # Get xywh, cfd, class
        true_cellbox = y_true[..., :4]
        true_obj = y_true[..., 4]
        true_cls = y_true[..., 5:]

        pred_cellbox1 = y_pred[..., :4]
        pred_obj1 = y_pred[..., 4]
        pred_cellbox2 = y_pred[..., 5:9]
        pred_obj2 = y_pred[..., 9]
        pred_cls = y_pred[..., 10:]

        # Convert cell box to corner bbox to compute iou
        true_corner_bbox = convert_cellbox_to_corner_bbox(true_cellbox, true_obj)
        pred_corner_bbox1 = convert_cellbox_to_corner_bbox(pred_cellbox1)
        pred_corner_bbox2 = convert_cellbox_to_corner_bbox(pred_cellbox2)

        # Compute iou
        iou_box1 = iou(pred_corner_bbox1, true_corner_bbox)
        iou_box2 = iou(pred_corner_bbox2, true_corner_bbox)

        # Get the highest iou
        ious = tf.stack([iou_box1, iou_box2], axis=-1)
        best_iou = tf.cast(tf.math.argmax(ious, axis=-1),
                           dtype=tf.float32)

        # Compute xy loss
        xy_loss = compute_xy_loss(true_cellbox[..., :2], pred_cellbox1[..., :2],
                                  pred_cellbox2[..., :2], true_obj, best_iou)

        # Compute wh loss
        wh_loss = compute_wh_loss(true_cellbox[..., 2:], pred_cellbox1[..., 2:],
                                  pred_cellbox2[..., 2:], true_obj, best_iou)

        # Compute obj. loss
        obj_loss = compute_obj_loss(true_obj, pred_obj1, pred_obj2, best_iou)

        # Compute no obj. loss
        no_obj_loss = compute_no_obj_loss(true_obj, pred_obj1, pred_obj2)

        # Compute class distribution loss
        cls_loss = compute_class_dist_loss(true_cls, pred_cls, true_obj)

        yolo_loss = (coord_weight * (xy_loss + wh_loss) + obj_loss +
                     noobj_weight * no_obj_loss + cls_loss)

        return yolo_loss
