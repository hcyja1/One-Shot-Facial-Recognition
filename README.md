# One-Shot-Facial-Recognition

Modern facial recognition technology plays a crucial role in various daily activities such as Authentication and Security. Moreover, these applications require accurate prediction over low number of sample availability. This repo demonstrates a Transfer Learning based Siamese Network to classify images for N-way one shot learning. This approach uses the Viola Jones algorithm as a preprocessing optimization step to focus and extract a region of interest (ROI) containing facial features within an image. The cropped region is then passed through a Siamese Network which uses VGG-Face as the twin networks, followed by calculating the covariance distance between extracted features, before being passed through a sigmoid activation function which produces the final output. The proposed method achieved an accuracy of 93.2% over 1344 test images for 100-way one shot learning.

Overall methodology:

![overall methodology](https://user-images.githubusercontent.com/73547478/209741863-89fb414d-82a9-41d2-9113-0ef5db6962ff.jpg)

VGG-Face Modified Architecture for Transfer Learning:
![VGGFACE](https://user-images.githubusercontent.com/73547478/209741774-4dd27234-ace5-4e4c-86b2-1fd31f085454.png)

Transfer Learning and Evaluation methodology:
![transfer learning and evaluation](https://user-images.githubusercontent.com/73547478/209741762-0d70aa9b-d020-49ee-b77b-9da81b846a84.jpg)



The full report can be read at https://drive.google.com/file/d/1qI4l6EZ_FRrRbkf8KAAKMlIHEEsUecDL/view?usp=sharing. 
