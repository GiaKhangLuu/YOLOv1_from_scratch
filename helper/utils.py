# ====================================
# IMPORT MODULES
# ====================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2


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


def convert_cellbox_to_xywh(cellbox, mask=None):
    """
    Convert tensor box (x_offset, y_offset, w, h) to
    bounding box (x_center, y_center, w, h)
    :param cellbox: Tensor box (batch_size, grid_size, grid_size, 4)
    :param mask: Tensor to determines which cell has obj
    :return bbox: Tensor box (batch_size, grid_size, grid_size, 4)
    """

    x_offset, y_offset = cellbox[..., 0], cellbox[..., 1]
    w_h = cellbox[..., 2:]

    num_w_cells = x_offset.shape[-1]
    num_h_cells = x_offset.shape[-2]

    # w_cell_indices: [[0, 1, 2, ...], [0, 1, 2, ...], ...]
    # Use w_cell_indices to convert x_offset of a particular grid cell
    # location to x_center
    w_cell_indices = np.array(range(num_w_cells))
    w_cell_indices = np.broadcast_to(w_cell_indices, x_offset.shape[-2:])

    # h_cell_indices: [[0, 0, 0, ...], [1, 1, 1, ...], [2, 2, 2, ...], ....]
    # Use h_cell_indices to convert y_offset of a particular grid cell
    # location to y_center
    h_cell_indices = np.array(range(num_h_cells))
    h_cell_indices = np.repeat(h_cell_indices, 7, 0).reshape(x_offset.shape[-2:])
    # h_cell_indices = np.broadcast_to(h_cell_indices, x_offset.shape)

    x_center = (x_offset + w_cell_indices) / num_w_cells
    y_center = (y_offset + h_cell_indices) / num_h_cells

    if mask is not None:
        x_center *= mask
        y_center *= mask

    x_y = tf.stack([x_center, y_center], axis=-1)

    bbox = tf.concat([x_y, w_h], axis=-1)

    return bbox


def convert_cellbox_to_corner_bbox(cellbox, mask=None):
    """
    Convert tensor box (x_offset, y_offset, w, h) to
    corner bounding box (x_min, y_min, x_max, y_max)
    :param cellbox: Tensor box (batch_size, grid_size, grid_size, 4)
    :param mask: Tensor to determines which cell has obj
    :return corner_bbox: Tensor box (batch_size, grid_size, grid_size, 4)
    """

    bbox = convert_cellbox_to_xywh(cellbox, mask)
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]

    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)

    corner_box = tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    return corner_box


def post_process_tensor_output(pred_tensor_output):
    """
    :param pred_tensor_output: Convert predicted tensor
    output to boxes, scores, classes
    :return (boxes, scores, classes, nums): Tuple contains
    boxes, scores, classes, nums
    """

    pred_box_1 = pred_tensor_output[..., :4]
    pred_cfd_1 = pred_tensor_output[..., 4]
    pred_box_2 = pred_tensor_output[..., 5:9]
    pred_cfd_2 = pred_tensor_output[..., 9]
    pred_cls_dist = pred_tensor_output[..., 10:]

    pred_corner_bbox_1 = convert_cellbox_to_corner_bbox(pred_box_1)
    pred_corner_bbox_2 = convert_cellbox_to_corner_bbox(pred_box_2)

    # To use combined_nms() method from TF we must change
    # [x1, y1, x2, y2] to [y1, x1, y2, x2]
    box1 = tf.reshape(tf.gather(pred_corner_bbox_1, [1, 0, 3, 2], axis=-1), shape=(-1, 7 * 7, 1, 4))
    box2 = tf.reshape(tf.gather(pred_corner_bbox_2, [1, 0, 3, 2], axis=-1), shape=(-1, 7 * 7, 1, 4))
    boxes = tf.concat([box1, box2], axis=1)

    scores1 = tf.reshape(tf.expand_dims(pred_cfd_1, axis=-1) * pred_cls_dist, shape=(-1, 7 * 7, 3))
    scores2 = tf.reshape(tf.expand_dims(pred_cfd_2, axis=-1) * pred_cls_dist, shape=(-1, 7 * 7, 3))
    scores = tf.concat([scores1, scores2], axis=1)

    boxes, scores, classes, nums = tf.image.combined_non_max_suppression(boxes, scores, max_output_size_per_class=10,
                                                                         max_total_size=49, iou_threshold=0.5,
                                                                         score_threshold=0.5)

    return boxes, scores, classes, nums


def draw_output(img, output, label_names):
    """
    Draw bounding box and label name on image
    :param img: Ndarray type image to draw
    :param output: Tuple of prediction (boxes, scores, classes, nums)
    :param label_names: List of label names
    :return img: Image has been drew
    """

    boxes, scores, classes, nums = output
    #img_wh = np.flip(img.shape[0:2])
    img_hw = img.shape[0:2]

    for i in range(nums):


        #x1y1 = tuple((boxes[i, :2].numpy() * img_wh).astype(np.int32))
        #x2y2 = tuple((boxes[i, 2:].numpy() * img_wh).astype(np.int32))
        x1y1 = tuple(np.flip((boxes[i, :2].numpy() * img_hw).astype(np.int32)))
        x2y2 = tuple(np.flip((boxes[i, 2:].numpy() * img_hw).astype(np.int32)))

        # Draw bounding box
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)

        # Append label
        img = cv2.putText(img, '{} {:.4f}'.format(label_names[int(classes[i])],
                                                  scores[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    return img


def prepare_image(filename, input_shape):
    """
    Resize image to expected dimension, and opt. apply some random transformation.
    :param filename: File name
    :param input_shape: Shape expected by the model (image will be resize accordingly)
    :return : 3D image array, pixel values from [0., 1.]
    """

    img = img_to_array(load_img(filename, target_size=input_shape)) / 255.

    return img
