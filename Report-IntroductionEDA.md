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

> We begin by importing the breed images from the specified folder.

```python
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images
```

> Obtain the absolute path to the image repo, load the images into an array. We then store the histogram intersection scores for each image, and create a new dataframe to store the features of the dog images. 

```python
GerShep_path = os.path.abspath("./Dog Images/n02106662-German_shepherd")
GerShep_imgs = load_images_from_folder(GerShep_path)
GerShep_Train_img = cv2.imread("./Dog Images/German_Shepherd.jpg")

Bulldog_path = os.path.abspath("./Dog Images/n02096585-Boston_bull")
Bulldog_imgs = load_images_from_folder(Bulldog_path)
Bulldog_Train_img = cv2.imread("./Dog Images/Bulldog.jpg")

GerShep_inter_sc = []
Bull_inter_sc = []

features_G = pd.DataFrame()
features_B = pd.DataFrame()
```

> We use the following breed photos as our training image (minimal backgound colours, clear view of dog), generating a series of histogram intersection scores (for both breeds) against the corresponding training image. 

Boston Bull Training Image          |  German Shepherd Training Image  
:-------------------------:|:-------------------------:
![Bull_train](/Images/Bulldog.jpg)  |  ![GerShep_train](/Images/German_Shepherd.jpg)

```python
trainingHist_G = cv2.calcHist([GerShep_Train_img], [0], None, [256], [0,256])
trainingHist_G = cv2.normalize(trainingHist_G, trainingHist_G).flatten()
trainingIntrResult_G = cv2.compareHist(trainingHist_G, trainingHist_G, cv2.HISTCMP_INTERSECT)

trainingHist_B = cv2.calcHist([Bulldog_Train_img], [0], None, [256], [0,256])
trainingHist_B = cv2.normalize(trainingHist_B, trainingHist_B).flatten()

trainingIntrResult_B = cv2.compareHist(trainingHist_B, trainingHist_B, cv2.HISTCMP_INTERSECT)

for i in range(0, len(GerShep_imgs)):
    
    hist1 = cv2.calcHist(GerShep_imgs[i],[0],None,[256],[0,256])
    hist1 = cv2.normalize(hist1,hist1).flatten()
    GerShep_inter_sc.append(cv2.compareHist(trainingHist_G, hist1, cv2.HISTCMP_INTERSECT))

for i in range(0, len(Bulldog_imgs)):

    hist2 = cv2.calcHist(Bulldog_imgs[i],[0],None,[256],[0,256])
    hist2 = cv2.normalize(hist2,hist2).flatten()
    Bull_inter_sc.append(cv2.compareHist(trainingHist_B, hist2, cv2.HISTCMP_INTERSECT))    
```



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
