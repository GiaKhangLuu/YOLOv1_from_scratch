# ==============================================
# IMPORT MODULES
# ==============================================

import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from xml.etree import ElementTree
import tensorflow as tf
from tqdm import tqdm
from functools import partial


# ==============================================
# CONSTANT DEFINITIONS
# ==============================================

class_names = ['apple', 'banana', 'orange']


# ==============================================
# FUNCTION DEFINITIONS
# ==============================================

def get_dataframe(file_dir):
    """
    Get the train/val/test dataframe which contains image
    file names and annotations files. If `phase = train',
    return train and val set
    :param file_dir: File directory to create dataframe
    :return file_df: Train or test dataframe
    """

    img_files = [os.path.join(file_dir, img_file) for img_file
                 in sorted(os.listdir(file_dir)) if img_file[-4:] == '.jpg']
    annot_files = [img_file[:-4] + '.xml' for img_file in img_files]

    img_file_series = pd.Series(img_files, name='Image_file')
    annot_file_series = pd.Series(annot_files, name='Annotation_file')
    file_df = pd.DataFrame(pd.concat([img_file_series, annot_file_series], axis=1))

    return file_df


def prepare_image(filename, input_shape):
    """
    Resize image to expected dimension, and opt. apply some random transformation.
    :param filename: File name
    :param input_shape: Shape expected by the model (image will be resize accordingly)
    :return : 3D image array, pixel values from [0., 1.]
    """

    img = img_to_array(load_img(filename, target_size=input_shape)) / 255.

    return img


def convert_to_xywh(bboxes):
    """
    Convert list of (xmin, ymin, xmax, ymax) to
    (x_center, y_center, box_width, box_height)
    :param bboxes: List of bounding boxes, each has 4
    values (xmin, ymin, xmax, ymax)
    :return boxes: List of bounding boxes, each has 4
    values (x_center, y_center, box_width, box_height)
    """

    boxes = list()
    for box in bboxes:
        xmin, ymin, xmax, ymax = box

        # Compute width and height of box
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Compute x, y center
        x_center = int(xmin + (box_width / 2))
        y_center = int(ymin + (box_height / 2))

        boxes.append((x_center, y_center, box_width, box_height))

    return boxes


def extract_annotation_file(filename):
    """
    Extract bounding boxes from an annotation file
    :param filename: Annotation file name
    :return boxes: List of bounding boxes in image, each box has
    4 values (x_center, y_center, box_width, box_height)
    :return classes: List of classes in image
    :return width: Width of image
    :return height: Height of image
    """

    # Load and parse the file
    tree = ElementTree.parse(filename)
    # Get the root of the document
    root = tree.getroot()
    boxes = list()
    classes = list()

    # Extract each bounding box
    for box in root.findall('.//object'):
        cls = class_names.index(box.find('name').text)
        xmin = int(box.find('bndbox/xmin').text)
        ymin = int(box.find('bndbox/ymin').text)
        xmax = int(box.find('bndbox/xmax').text)
        ymax = int(box.find('bndbox/ymax').text)
        coors = (xmin, ymin, xmax, ymax)
        boxes.append(coors)
        classes.append(cls)

    boxes = convert_to_xywh(boxes)

    # Get width and height of an image
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)

    # Some annotation files have set width and height by 0,
    # so we need to load image and get it width and height
    if (width == 0) or (height == 0):
        img = load_img(filename[:-4] + '.jpg')
        width, height = img.width, img.height

    return boxes, classes, width, height


def convert_bboxes_to_tensor(bboxes, classes, img_width,
                             img_height, grid_size=7):
    """
    Convert list of bounding boxes to tensor target
    :param bboxes: List of bounding boxes in image, each box has
    4 values (x_center, y_center, box_width, box_height)
    :param classes: List of class in image
    :param img_width: Image's width
    :param img_height: Image's height
    :param grid_size: Grid size
    :return target: Target tensor (grid_size x grid_size x (5 + num_classes))
    """

    num_classes = len(class_names)
    target = np.zeros(shape=(grid_size, grid_size, 5 + num_classes), dtype=np.float32)

    for idx, bbox in enumerate(bboxes):
        x_center, y_center, width, height = bbox

        # Compute size of each cell in grid
        cell_w, cell_h = img_width / grid_size, img_height / grid_size

        # Determine cell i, j of bounding box
        i, j = int(y_center / cell_h), int(x_center / cell_w)

        # Compute value of x_center and y_center in cell
        x, y = (x_center / cell_w) - j, (y_center / cell_h) - i

        # Normalize width and height of bounding box
        w_norm, h_norm = width / img_width, height / img_height

        # Add bounding box to tensor
        # Set x, y, w, h
        target[i, j, :4] += (x, y, w_norm, h_norm)
        # Set obj score
        target[i, j, 4] = 1.
        # Set class dist.
        target[i, j, 5 + classes[idx]] = 1.

    return target


def load_dataset(dataframe, input_shape, grid_size=7):
    """
    Load img and target tensor
    :param dataframe: Dataframe contains img files and annotation files
    :param input_shape: Shape expected by the model (image will be resize accordingly)
    :param grid_size: Grid size
    :return dataset: Iterable dataset
    """

    imgs, targets = list(), list()

    for _, row in tqdm(dataframe.iterrows()):
        img = prepare_image(row.Image_file, input_shape)
        target = extract_annotation_file(row.Annotation_file)
        target = convert_bboxes_to_tensor(*target, grid_size)
        imgs.append(img)
        targets.append(target)

    imgs = np.array(imgs)
    targets = np.array(targets)

    dataset = tf.data.Dataset.from_tensor_slices((imgs, targets))
    return dataset


def _apply_augmentation(image, target, seed=None):
    """
    Apply random brightness and saturation on image
    :param image: Image to augment
    :param target: Target tensor
    :param seed: Seed for random operation
    :return : Processed data
    """

    # Random bright & saturation change
    image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)

    # Keeping pixel values in check
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image, target


def load_dataset_from_df(dataframe, batch_size=32, num_epochs=None, shuffle=False,
                         input_shape=(448, 448, 3), grid_size=7, augment=False,
                         seed=None):
    """
    Instantiate dataset
    :param dataframe: Dataframe contains img files and annotation files
    :param batch_size: Batch size
    :param num_epochs: Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle: Flag to shuffle the dataset (if True)
    :param input_shape: Shape of the processed image
    :param grid_size: Grid size
    :param augment: Flag to apply some random augmentations to the image
    :param seed: Random seed for operation
    :return : Iterable dataset
    """

    apply_augmentation = partial(_apply_augmentation, seed=seed)

    dataset = load_dataset(dataframe, input_shape, grid_size)
    dataset = dataset.repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(1000, seed)
    if augment:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


