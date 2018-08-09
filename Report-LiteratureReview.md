---
title: Literature Review and Related Work
---

## Contents
{:.no_toc}
*  
{: toc}



### 1. “Dog breed classification using part localization” - Jiongxin Liu, Anglo Kanazawa, David Jacobs, Peter Belhumeur, Columbia University & University of Maryland (2012)

This Paper introduces a process for fine grain feature detection, in successfully classifying dog breeds with 67% accuracy. The paper’s approach leverages automatic face detection, localization of eyes and nose relative to the dog’s face, attempts to align the face and extract a grayscale scale-invariant feature transform (SIFT). Furthermore, the paper’s approach leverages the use of a color histogram to extract additional SIFT features.

In reviewing this paper, the author outlines that minimal preprocessing was performed on 8,351 photos, with the exception that both of a dog's eyes must be visible in relation to their face. The paper did not disclose if any photos we purely
colour, black and white, or a mix.

### 2. “Transfer Learning for Image Classification of various dog breeds” Pratik Devikar, IJARCET (2016)

Paper proposes the use of transfer learning in successfully (96% accuracy) identifying dog breeds from large image datasets. Rather than recreating a new classification model from scratch, the author leverages and retrains Google’s Inception model V3 to correctly classify an initial suite of 275 dog images belonging to a set of 11 unique dog categories.

Devikar suggests that most transfer learning methodologies rely on both new dataset size, as well as data content similarity. The author outlines the process of fine tuning Convolutional Neural Networks (CNN) for dog classification required, in which a new datasets of smaller comparative size versus the original training set was used. Furthermore, the content similarity of new training data compared against the original training set, was significantly different (in the particular case of dog classification). As such, Devikar claims that only a linear classifier should be trained.

Devikar provides insight into a common drawback with leveraging neural networks, in which their flexibility has the potential tendency of overfitting data (they learn features and noise well). To address this issue the author describes their process of augmenting the data by simply rotating the images 90, 180, 270 degrees. The author also suggests that popular augmentation techniques of horizontally flipping, random cropping and color jittering can be used.

### 3. “Indexing via Color Histograms”, Michael J. Swain, Dana H. Ballard, Department of Computer Science, University of Rochester

Swain et. al. outline two issues with image identification tasks, identifying an object in a known location, and locating a known object. The authors propose that these issues can be resolved in real time through the use of color histograms. Furthermore, they suggest that color should be considered as a primary feature in most computer vision tasks.

The paper describes a color histogram as "counting the number of times each color occurs in the image array". Furthermore, it outlines that color histograms are invariant to image translation and rotation, and only fluctuate slowly under angle changes.

Swain et. al. further describe a Histogram Intersection algorithm to determine model to image fit, as the match between the image color histogram with the histograms of each model in the database. The higher the Histogram Intersection match value, the better fit the image is with the compared model.

The Histogram Intersection match value is unlikely to be affected by background pixel values, as the background image and model pixels are unlikely to be equal. This resolves the need for object/background isolation, which could prove computationally and time intensive, thus reducing dataset preprocessing.

### 4. “Dog Breed Classification using Deep Learning: hands-on approach”, Kirill Panarin, towardsdatascience.com

Panarin extends the notion of transfer learning in this article, by providing a demonstration of the processing in action. Panarin outlines the process of combining transfer learning with neural networks by feeding an image into a inception model, the output then going through several fully connected layers, which finally a suite of probabilities are assigned to each class.

Panarin also provides insight into leading contenders for misclassification amongst dog breeds, with the Yorkshire Terrier and Silky Terrier as common outliers.

Overall the article provides justification of transfer learning, as a means to offset small datasets, so long as the user has access to pre-trained deep neural networks and modern machine learning libraries (such as TensorFlow)
