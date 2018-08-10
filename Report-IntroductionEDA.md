---
title: Introduction and EDA
---

## Contents
{:.no_toc}
*  
{: toc}



## 1. Introduction and Description of Data

### 1) 

### 2) 

## 2. Exploratory Data Analysis

### 1) Histogram Intersections

In our preliminary exploratory data analysis, we took the average value for each pixel for the Boston Bulldog and German Shepherd (each with a “noise-less” background), and plotted these values onto a histogram. We noted that there was a possibility of classification with color as one of the criteria. Our subesquent analysis into scoring the histogram intersections on the fur color of German Shepherd and Boston Bull breed images indicated that correctly identifying fellow dog breeds with just fur color alone was possible, however with such features came the potential for model inaccuracy.

In plotting color histograms for both German Shepherd and Boston Bull breeds, we took in an array of source images, each with the same depth, provided a singular grayscale channel for the according image, and produced a histogram using the OpenCV library method (calcHist). Using the compareHist function, with CV_COMP_INTERSECT method, we then compared these histograms againsts a training histogram to evaluate their score. Additionally, our group further modified the comparison algorithm to first normalize the histogram before performing a comparison. 

Boston Bull Knn R2           |  German Shepherd Knn R2
:-------------------------:|:-------------------------:
![Bull_knn_r2](/Images/Bull_knn_r2.png)  |  ![GerShep_knn_r2](/Images/GerShep_knn_r2.png)


Boston Bull Knn         |  German Shepherd Knn 
:-------------------------:|:-------------------------:
![Bull_Hist_Intersect_knn](/Images/Bull_Hist_Intersect_knn.png)  |  ![GerShep_Hist_Intersect_knn](/Images/GerShep_Hist_Intersect_knn.png)


| Boston Bull & German Shepherd Histogram  |
|---|
|![GerBull_Hists](/Images/GerBull_Hists.png)|

Boston Bull Intersection Scores           |  German Shepherd Intersection Scores
:-------------------------:|:-------------------------:
![Bull_Hist_Intersect_Scores](/Images/Bull_Hist_Intersect_Scores.png)  |  ![GerShep_Hist_Intersect_Scores](/Images/GerShep_Hist_Intersect_Scores.png)

Our model using a normalized score, such that a value of 1 indicates a perfect match between the two color histograms, while 0 suggests an inability to produce a match. In presenting our model findings, our group had not yet utilize a combination of PCA and fur color as potential classification features, this was later added to our final model. Finally, we can see that normalized color histogram intersection matching demonstrates moderately accurate intersection scoring for dog breed classification, with Boston Bull breeds out performing in comparison to German Shepherds.

### 2)
