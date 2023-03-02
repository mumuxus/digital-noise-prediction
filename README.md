# digital-noise-prediction
Investigating the impacts of added white Gaussian noise on digital images and developing machine learning models to predict the level of noise in an image set using Python, OpenCV, and various other libraries.


## Technologies Used
- Python
- OpenCV
- Matplotlib
- NumPy
- pandas
- scikit-learn

## Introduction 
This project aims to explore the complexities of identifying and removing noise in digital images. The report provides a thorough understanding of the concepts and techniques involved in detecting and removing noise in images. The project implements five different models, three of which deal with identifying noise in images and the other two with removing noise from images.

## Dataset
The dataset used in this project comprises of 7129 grayscale images of various landscapes. The images are processed to make sure that each image is 150x150 pixels, and any corrupted data is removed. Synthetic Gaussian, Salt and Pepper, and Poisson noise are added to the dataset using three different functions, with a 80% chance that each image will have some form of added noise. The dataset is then normalized and split into an 85/15 train/test split.

## Models
1. Binary SVM Classifier: This model uses a support vector machine (SVM) with a radial basis function (RBF) kernel to perform a binary classification on the dataset. The goal is to predict if an image is noisy or clear.
2. Multiclass SVM Classifier: This model is similar to the binary SVM classifier but the labels are changed to the eight different noise cases, instead of a binary label. The classifier uses a one vs. rest approach to match the noisy image to the corresponding noise case.
3. Simplified Multiclass SVM Classifier: This model is similar to the multiclass SVM classifier but the number of classes is reduced to four (no noise, Gaussian noise, salt and pepper noise, Poisson noise).
4. Deep Learning with Auto Encoders: This model uses a convolutional neural network (CNN) with an autoencoder architecture to remove noise from the images. The network is trained over 100 epochs, and the results show that the network was too simplified for the complex task at hand.
5. Deep Learning with Auto Encoders (Simplified): This model is similar to the previous deep learning model but the dataset is simplified to only contain white Gaussian noise. The results show that the network still produced poor results.

## Results and Conclusion 
In this research project, I delved into the intricacies of noise in digital images and gained a comprehensive understanding of the subject. I started by exploring the difference between real and synthetic noise and attempted to mimic real noise in the code. Then, I applied support vector machines to detect noise in the dataset. Although I found it easy to identify if an image was noisy, it proved to be challenging to label different types of noise.

Next, I delved into the world of deep learning, specifically convolutional neural networks and auto encoders. My goal was to use these tools to denoise the images. Although my first attempt using a simple auto encoder network was not successful, it gave me valuable insights into the importance of network sophistication and dimensionality reduction in image reconstruction.

The first model, a binary classifier using support vector machines, was successful and could be used as a preprocessing step in a larger algorithm to identify noisy images in a dataset. Moving forward, I aim to improve upon these models and bring myself closer to detecting and estimating the noise level of an image with higher accuracy. I also plan to test these models on different datasets, including colored and medical images, to expand the scope of my research.

In conclusion, this research project was a valuable learning experience that gave us a deeper appreciation for the complexities of digital noise and motivated me to continue exploring the field.
