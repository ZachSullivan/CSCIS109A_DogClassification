---
title: Overview:
---


## CSCI S 109A Dog Classification Final Project
#### Group #17 Zachary Sullivan, Yuil Ahn, Samuel Akwei-Sekyere 

### Motivation:
Identification of dog breeds is a difficult task. While experts and general public have difficulty in correctly identifying dog breeds, dog breeds in of themselves, have a wide variety of mixes and variations and as such are regulated to strict breed and group classifications. For instance, the Fédération Cynologique Internationale (FCI), classifies sheepdogs and cattle dogs to include German and Belgian shepherds, while excluding the Swiss Mountain Cattle dog.

In the case of our project, Stanford University’s dog breed image repository dataset required the incorporation of image processing techniques for pre-processing, unsupervised learning for feature extraction and supervised learning algorithms for classification. 

### Problem Statement:

Using the dataset provided by [Stanford University’s dog breed image repository](http://vision.stanford.edu/aditya86/ImageNetDogs/), we will implement and evaluate models for classifying pure dog breeds into various classes.
In order to mitigate risk in successfully implementing a classification algorithm, our initial algorithm will attempt to distinguish between German Shepherd and Boston Bull breeds. 

These breeds were chosen on the basis of their comparatively distinct visual qualities (for instance fur color). In order to classify the preliminary German Shepherd and Boston Bull breeds, we will be taking advantage of the following parameters:

1. Color (Through Histogram Intersection matching)
2. PCA features

As such, we will attempt to classify these breeds based on each image's pixel value. Furthermore, our group will leverage a suite of classification models based on material covered in class. Specificaly, we will utilize a K-Nearest Neighbors classifier, an AdaBoost Classification model, a Decision Tree and Random Forest classifier, and finally attempt a Logistic regression meta-model (based on the performance of all prior models).
