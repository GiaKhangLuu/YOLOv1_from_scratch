# ===================================================
# IMPORT MODULES
# ===================================================

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, BatchNormalization, Activation,
                                     LeakyReLU, Dropout, InputLayer, MaxPooling2D)
from tensorflow.keras.models import Sequential, Model

# ===================================================
#  CONSTANT DEFINITION
# ===================================================

input_shape = (224, 224, 3)

# YOLO backbone architecture based on original paper
yolo_backbone_architecture = [(64, 7, 2, 'same'), 'M',
                              (192, 3, 1, 'same'), 'M',
                              (128, 1, 1, 'valid'),
                              [(128, 256), 1],
                              [(256, 512), 1], 'M',
                              [(256, 512), 4],
                              [(512, 1024), 1], 'M',
                              [(512, 1024), 2]]


# ===================================================
#  CLASS DEFINITION
# ===================================================

class ConvWithBatchNorm(Conv2D):
    """Conv layer with batch norm and leaky relu"""

    def __init__(self, filters=64, kernel_size=3, strides=1, padding='same',
                 activation=LeakyReLU(alpha=0.1), kernel_regularizer=None,
                 name='conv', **kwargs):
        """
        Initialize the layer
        :param filters: Number of filters
        :param kernel_size: Kernel size
        :param strides: Convolution strides
        :param padding: Convolution padding
        :param activation: Activation function (name or callable)
        :param kernel_regularizer: Kernel regularization method
        :param name: Name suffix for the sub_layers
        :param kwargs: Optional parameters of Conv2D
        """

        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, activation=None,
                         kernel_regularizer=kernel_regularizer, name=name, **kwargs)
        self.batch_norm = BatchNormalization(name=name + '_bn')
        self.activation = Activation(activation, name=name + '_act')

    def call(self, inputs):
        """
        Call the layer
        :param inputs: Input tensor
        :return: Output tensor
        """

        x = super().call(inputs)
        return self.activation(self.batch_norm(x))


class BottleNeckBlock(Sequential):
    """Block of 1x1 reduction layers followed by 3x3 conv. layer"""

    def __init__(self, filters, repetitions, name='bottleneck_block', **kwargs):
        """
        Initialize the layers
        :param filters: Tuple of filters
        :param repetitions: Number of times the block should be repeated inside
        :param kwargs: Optional parameters of Conv2D
        """

        filters_1x1 = filters[0]
        filters_3x3 = filters[1]
        model = []
        for i in range(repetitions):
            model += [ConvWithBatchNorm(filters=filters_1x1, kernel_size=1,
                                        strides=1, padding='valid',
                                        name='conv_1x1_{}'.format(i + 1), **kwargs)]
            model += [ConvWithBatchNorm(filters=filters_3x3, kernel_size=3,
                                        strides=1, padding='same',
                                        name='conv_3x3_{}'.format(i + 1), **kwargs)]

        super().__init__(model, name=name)


class YoloBackbone(Sequential):
    """YOLO backbone extract feature from the input"""

    def __init__(self, input_shape=input_shape,
                 backbone_config=yolo_backbone_architecture,
                 name='YOLO_Backbone'):
        """
        Initialize the layers
        :param input_shape: Input shape
        :param backbone_config: List of configurations of YOLO backbone
        :param name: Name suffix for the sublayer
        """

        model = [InputLayer(input_shape=input_shape)]

        for i, config in enumerate(backbone_config):
            if type(config) == tuple:
                filters, kernel_size, strides, padding = config
                model += [ConvWithBatchNorm(filters, kernel_size, strides, padding,
                                            name='backbone_conv_{}'.format(i + 1))]
            elif type(config) == str:
                model += [MaxPooling2D(pool_size=2, strides=2, padding='same',
                                       name='backbone_max_pooling_{}'.format(i + 1))]
            elif type(config) == list:
                filters, repetition = config
                model += [BottleNeckBlock(filters, repetition,
                                          name='backbone_bottleneck_block_{}'.format(i + 1))]

        super(YoloBackbone, self).__init__(model, name=name)


class YoloOutput(Sequential):
    """YOLO last convolution and FC layers to produce prediction"""

    def __init__(self, fv_shape, grid_size=7,
                 num_boxes=2, num_classes=20, name='YOLO_Output'):
        """
        Initialize the layers
        :param fv_shape: Feature volume input from the last conv. backbone
        :param grid_size: Grid size
        :param num_boxes: Number of bounding boxes
        :param num_classes: Number of classes
        """

        S, B, C = grid_size, num_boxes, num_classes

        yolo_output = [ConvWithBatchNorm(filters=1024, kernel_size=3, strides=1,
                                         padding='same', name='output_conv_1', input_shape=fv_shape),
                       ConvWithBatchNorm(filters=1024, kernel_size=3, strides=2,
                                         padding='same', name='output_conv_2'),
                       ConvWithBatchNorm(filters=1024, kernel_size=3, strides=1,
                                         padding='same', name='output_conv_3'),
                       ConvWithBatchNorm(filters=1024, kernel_size=3, strides=1,
                                         padding='same', name='output_conv_4'),
                       Flatten(),
                       Dense(units=4096, activation=LeakyReLU(alpha=0.1), name='output_fc_1'),
                       Dropout(rate=0.5, name='dropout'),
                       Dense(units=S * S * (B * 5 + C), name='prediction')]

        super().__init__(yolo_output, name=name)


class YoloV1(Model):
    """End-to-end YOLO network"""

    def __init__(self, input_shape, grid_size=7, num_boxes=2, num_classes=20,
                 backbone_config=yolo_backbone_architecture, name='YOLO_V1'):
        """
        Initializer YOLO_v1
        :param input_shape: Input shape
        :param grid_size: Grid size to split
        :param num_boxes: Number of bounding boxes
        :param num_classes: Number of classes
        :param backbone_config: List of configurations of YOLO backbone
        :param name: Model's name
        """

        super().__init__(name=name)
        self.yolo_backbone = YoloBackbone(input_shape=input_shape,
                                          backbone_config=backbone_config)

        backbone_output = self.yolo_backbone.output_shape[1:]

        self.yolo_output = YoloOutput(fv_shape=backbone_output, grid_size=grid_size,
                                      num_boxes=num_boxes, num_classes=num_classes)

    def call(self, inputs):
        """
        Call the model
        :param inputs: Input tensor
        :return : Output tensor
        """

        return self.yolo_output(self.yolo_backbone(inputs))
