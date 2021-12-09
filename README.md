# person_detection_tflm
This project is an attempt to create a person detection TFlite micro model for embedded microcontrollers. The model was created in TensorFlow Keras, converted to .tflite model with 8bit quantization and than converted to C char array model used with TFlite micro library. The scripts for creating the dataset, training the .h5 model, quantizing and converting to .tflite and evaluating the model are provided in this repository. The model is converted to C char array using "xxd" linux command tool.
The model is still too large for deployment to most available MCU devices with over 2MB in size. It can be reduced by retraining the model for smaller images or changing the model architecture.
The custom made dataset is composed of 25k images from COCO and OIDv4 datasets. The images from these datasets were cropped to square size depending on where the person was in the original image, using provided annotations. Cropped images were than resized to 120\*120 size and those are provided here under [person](Dataset/person_120x120) and [not_person](Dataset/not_person_120x120) directories. The [dataset_array](Dataset/dataset_array.pickle) pickle file contains gray-scale images in Numpy array of shape (25000,120,120).
## Accuracy and loss
<img src="https://github.com/MarioGavran/person_detection_tflm/blob/master/Training_plots/accuracy_128b-5e-021220210128.png" width="500"> <img src="https://github.com/MarioGavran/person_detection_tflm/blob/master/Training_plots/loss_128b-5e-021220210128.png" width="500">

## Examples of the dataset
* Every image has two titles. The bottom title is the label and the top title is the result of the inference.
* The model shows 83% accuracy on whole dataset
* The model shows about 75% accuracy on validation dataset that it has never seen before.
<img src="https://github.com/MarioGavran/person_detection_tflm/blob/master/images/Figure_1.png">
<img src="https://github.com/MarioGavran/person_detection_tflm/blob/master/images/Figure_2.png">
<img src="https://github.com/MarioGavran/person_detection_tflm/blob/master/images/Figure_3.png">
