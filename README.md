Title: Camera Based Indoor Object Detection and Distance Estimation Framework
Project 2 Paper: https://ieeexplore.ieee.org/document/10294458
Online Code: https://github.com/paul-pias/Object-Detection-and-Distance-Measurement
Data Set:
For this object detection project, we collected a total of 200 images featuring three classes: table,
chair, and trash bin. These images were taken within the university premises, ensuring diversity in
terms of lighting, angles, and distances. The dataset is organized as follows:
• Classes:
1. Table
2. Chair
3. Trash bin
   
• Dataset Split: The images are split into three parts:
o Training Set: 80% (160 images)
o Validation Set: 10% (20 images)
o Test Set: 10% (20 images)
• Annotations: Each image is annotated with bounding boxes around the objects, specifying
the location of each object in the image. The annotations are stored in a text file in the
following format:
▪ image_filename x_min, y_min, x_max, y_max, class_id
▪ image_filename: The name of the image file
▪ x_min, y_min: The coordinates of the top-left corner of the bounding box
▪ x_max, y_max: The coordinates of the bottom-right corner of the bounding
box
▪ class_id: A number representing the class of the object (0 for table, 1 for chair,
and 2 for trash bin)
This structure allows for efficient training, validation, and testing of the object detection model.
Data set link: https://unhnewhaven
my.sharepoint.com/:f:/g/personal/rmanu4_unh_newhaven_edu/EnVqtmN_Ua1AoCK6Hu82cF0BK
I9m30l9ErpZRPIltFreKA?e=hnLvMI
Data Normalization and Preprocessing:
The images in the dataset are preprocessed using several steps to prepare them for training:
Normalization: The pixel values of the images are normalized to the range [0, 1] by dividing the
pixel values by 255. This is done after converting the images to a PyTorch tensor. The normalization
step ensures that the pixel values are in a standardized range, making the training process more stable
and efficient.
o Example: img_ = torch.from_numpy(img_).float().div(255.0)
Resizing: The images are resized to a fixed dimension (inp_dim x inp_dim, where inp_dim is
typically 416 or 608 for YOLOv4) using the letterbox_image function to ensure consistency in input
size for the model.
Transformation: The images are transformed into tensors and reordered to the format C x H x W,
as required by PyTorch models. The transformation ensures the images are in the correct format for
feeding into the YOLOv4 model.
Data Augmentation and Transformation:
Data augmentation techniques are often used in object detection to improve the model’s robustness.
While not explicitly detailed in the provided code, these transformations are typically applied during
training to artificially increase the dataset size and variability. Some common transformations
include:
• Random Cropping: Randomly cropping portions of the image while maintaining object
visibility.
• Flipping: Horizontally flipping the images to improve invariance to object orientation.
• Color Jitter: Modifying the brightness, contrast, and saturation to simulate different lighting
conditions.
These transformations help the model generalize better by exposing it to various scenarios during
training
YOLOv4 Model Description:
The YOLOv4 (You Only Look Once version 4) model is a state-of-the-art object detection
framework designed for real-time applications. YOLOv4 is known for its speed and accuracy,
making it highly suitable for object detection tasks like the one described here. Below are the key
components of the YOLOv4 model:
1. Backbone: YOLOv4 uses CSPDarknet53 as its backbone, which is responsible for
extracting features from the input images. This feature extraction process is crucial for
detecting objects, especially small or distant ones.
2. Neck: The model uses a PANet (Path Aggregation Network) as its neck, which helps
aggregate features from multiple scales, improving the model’s ability to detect objects of
various sizes, from small objects to large ones.
3. Head: The head of the model consists of detection layers that predict:
o Bounding box coordinates: The position and size of the detected objects.
o Class labels: The categories to which the detected objects belong (table, chair, trash
bin).
o Objectness score: A measure of how likely it is that the detected box contains an
object.
4. Anchor Boxes: YOLOv4 uses predefined anchor boxes to predict bounding boxes around
objects. These anchor boxes are learned during training, helping the model effectively detect
objects of different shapes and sizes.
5. Non-Maximum Suppression (NMS): After detecting multiple bounding boxes for the same
object, YOLOv4 applies NMS to remove overlapping boxes, retaining only the most
confident predictions.
Loss Function:
In the Object Detection and Distance Measurement project, I used Mean Average Precision (mAP)
as the loss function to optimize the performance of the model. mAP is an evaluation metric that
measures the accuracy of both object classification and localization by calculating the intersection
over union (IoU) between predicted and ground truth bounding boxes. By treating mAP as a loss
function, the model was trained to maximize the overlap between predicted and ground truth boxes,
minimizing misclassifications and improving the localization of objects.
This approach allowed the model to directly optimize its predictions, focusing on both classification
accuracy and bounding box precision. mAP evaluates the model's performance across different
classes and recall values, making it a comprehensive metric for object detection tasks. By using mAP
as a loss function, the model was able to learn more effectively and improve its ability to detect and
measure the distance to objects in real-world scenarios.
