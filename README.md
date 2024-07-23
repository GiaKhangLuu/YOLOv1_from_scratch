# **Introduction** 

Building YOLOv1 from scratch based on Tensorflow/Keras framework. The fruits dataset contains three categories like apple, banana and orange. This model detects these fruits on the image and draw bounding on it. The dataset contains 240 training images with four categories (apple, banana, orange, mixed), and the test set contains 60 images. 

YOLOv1 paper: https://arxiv.org/abs/1506.02640

## **<span>&#x1F195;</span> YOLOv2 is released <span>&#x1F195;</span>**

The YOLOv2 built from scratch is [here](https://github.com/GiaKhangLuu/YOLOv2_from_scratch)!!!. This new repository helps us to gain a deeper understand about how anchor boxes work and the way to define them.

# **Method and technique used in this project**

YOLOv1 architecture:

![YOLO-v1-network-structure-Yolo-v2-Tiny-has-fewer-parameters-than-Yolo-v1-Its-network](https://user-images.githubusercontent.com/64302789/137927802-dc25e1b0-9360-446a-9f83-0b3facef9071.jpg)

YOLOv1 loss:

![fe9kH](https://user-images.githubusercontent.com/64302789/137928076-b61d4ef7-9eb7-4a6a-82eb-ac0fe0f3f02a.png)

The model and loss function was built according to the paper. The model contains one Dropout layer with rate = 0.5. Training images was random changed brightness with `max_delta` = 1 and saturation with `lower` = 0.5 and `upper` = 1.5. 

Training set contains 240 images and 240 annotation (.xml) files, testing set contains 60 images and 60 annotation (.xml) files.

The model was trained approximately 10000 epochs and that lasts total over 4 days.

# **Prediction example**

**Detect apple**

![Screen Shot 2021-10-19 at 21 32 02](https://user-images.githubusercontent.com/64302789/137931769-86b7d88a-02b6-4895-9381-0f2915a9b62a.png)

**Detect banana**

![Screen Shot 2021-10-19 at 21 33 45](https://user-images.githubusercontent.com/64302789/137932094-03d7023c-5652-4755-a6cb-0139094dfc5e.png)

**Detect orange**

![Screen Shot 2021-10-19 at 21 42 42](https://user-images.githubusercontent.com/64302789/137933816-8a085e6b-a77f-4099-9e05-ae7bd5b3e747.png)

**Detect mixed**

![Screen Shot 2021-10-19 at 21 43 58](https://user-images.githubusercontent.com/64302789/137934023-ea88f5d5-f75e-44a6-861f-96d756c97098.png)






