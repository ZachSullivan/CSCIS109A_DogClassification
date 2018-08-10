---
title: Exploratory Data Analysis
---

## Contents
{:.no_toc}
*  
{: toc}

## 1. Exploratory Data Analysis

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
    
GerShp_inter_scNorm = [float(i)/trainingIntrResult_G for i in GerShep_inter_sc]
Bull_inter_scNorm = [float(i)/trainingIntrResult_B for i in Bull_inter_sc]

features_G['HISTCMP_INTERSECT'] = GerShep_inter_sc
features_G['HISTCMP_INTERSECT_Norm'] = GerShp_inter_scNorm
features_G['Image_Index'] = range(0, len(GerShep_imgs))

features_B['HISTCMP_INTERSECT'] = Bull_inter_sc
features_B['HISTCMP_INTERSECT_Norm'] = Bull_inter_scNorm
features_B['Image_Index'] = range(0, len(Bulldog_imgs))

G_intersec_norm_median= [np.median(i) for i in GerShp_inter_scNorm]
B_intersec_norm_median= [np.median(i) for i in Bull_inter_scNorm]
```

> Create a KNN model, fitting the model to training data and calculating the corresponding R^2 score

```python
X_trainG = features_G[['Image_Index']]
y_trainG = features_G[['HISTCMP_INTERSECT_Norm']]

X_trainB = features_B[['Image_Index']]
y_trainB = features_B[['HISTCMP_INTERSECT_Norm']]

def generate_KnnReg_r2 (X_train, y_train, k_max):
    train_Scores = [] 

    for k in range(1, k_max):
        knnreg = KNeighborsRegressor(n_neighbors=k) # Create KNN model
        knnreg.fit(X_train, y_train) # Fit the model to training data
        score_train = knnreg.score(X_train, y_train) # Calculate R^2 score
        train_Scores.append(score_train)

    return train_Scores

def generate_KnnReg (X_train, y_train, k):
    knnreg = KNeighborsRegressor(n_neighbors=k)
    knnreg.fit(X_train, y_train) # Fit the model to training data
    
    xgrid = np.linspace(np.min(X_train), np.max(X_train), 100)
    prediction = knnreg.predict(xgrid.reshape(100,1))
    
    return prediction
```

```python
gScores = generate_KnnReg_r2(X_trainG, y_trainG, 15)

# Plot the R2 score for each knn
plt.plot(range(1, 15), gScores,'o-')

plt.xlabel('k')
plt.ylabel('R-Squared')
plt.title("KNN Intersection Scores for German Shepherd Images")
plt.savefig('GerShep_knn_r2.png')
plt.show()

prediction = generate_KnnReg(X_trainG, y_trainG, 10)

plt.plot(X_trainG, y_trainG, 'o', label="Intersection Scores", alpha=0.5)
plt.plot(xgrid, prediction, label="10-NN", color='red')

plt.legend()
plt.xlabel("Image #")
plt.ylabel("Intersection Score")
plt.title("KNN Intersection Scores for German Shepherd Images")
plt.savefig('GerShep_Hist_Intersect_knn.png')
plt.show()

bScores = generate_KnnReg_r2(X_trainB, y_trainB, 15)

# Plot the R2 score for each knn
plt.plot(range(1, 15), bScores,'o-')

plt.xlabel('k')
plt.ylabel('R-Squared')
plt.title("KNN Intersection Scores for Boston Bull Images")
plt.savefig('Bull_knn_r2.png')
plt.show()

prediction = generate_KnnReg(X_trainB, y_trainB, 10)

plt.plot(X_trainB, y_trainB, 'o', label="Intersection Scores", alpha=0.5)
plt.plot(xgrid, prediction, label="10-NN", color='red')

plt.legend()
plt.xlabel("Image #")
plt.ylabel("Intersection Score")
plt.title("KNN Intersection Scores for Boston Bull Images")
plt.savefig('Bull_Hist_Intersect_knn.png')

plt.show()
```

Boston Bull Knn R2           |  German Shepherd Knn R2
:-------------------------:|:-------------------------:
![Bull_knn_r2](/Images/Bull_knn_r2.png)  |  ![GerShep_knn_r2](/Images/GerShep_knn_r2.png)


Boston Bull Knn         |  German Shepherd Knn 
:-------------------------:|:-------------------------:
![Bull_Hist_Intersect_knn](/Images/Bull_Hist_Intersect_knn.png)  |  ![GerShep_Hist_Intersect_knn](/Images/GerShep_Hist_Intersect_knn.png)

> Using a normalized score, our model indicates that a value of 1 is a perfect match between the two color histograms, while 0 suggests an inability to produce a match. In presenting our model findings, our group had not yet utilize a combination of PCA and fur color as potential classification features, this was later added to our final model. Finally, we can see that normalized color histogram intersection matching demonstrates moderately accurate intersection scoring for dog breed classification, with Boston Bull breeds out performing in comparison to German Shepherds.

```python
plt.hist( features_G['HISTCMP_INTERSECT_Norm'], bins=15, color='#539caf', alpha=1, label="German Shepherd Images" )
plt.hist( features_B['HISTCMP_INTERSECT_Norm'], bins=15, color='#7663b0', alpha=0.7, label="Boston Bull Images" )
plt.ylabel("Frequency")
plt.xlabel("Intersection Score")
plt.title("Color Histogram Intersection Scores for German Shepherd and Boston Bull Images")
plt.legend()
plt.savefig('GerBull_Hists.png')
plt.show()

ax = sns.distplot(features_G['HISTCMP_INTERSECT_Norm'], hist=True, kde=True, bins = 25, color = '#539caf')
ax.set(xlabel = 'Histogram Intersection Score (1.0 = Perfect Fit)', ylabel = 'Frequency')
ax.set_title("Frequency of German Shepherd Color Histogram Intersection Scores")
fig = ax.get_figure()
fig.savefig("GerShep_Hist_Intersect_Scores.png")
plt.show()

ax = sns.distplot(features_B['HISTCMP_INTERSECT_Norm'], hist=True, kde=True, bins = 25, color = '#7663b0')
ax.set(xlabel = 'Histogram Intersection Score (1.0 = Perfect Fit)', ylabel = 'Frequency')
ax.set_title("Frequency of Boston Bull Color Histogram Intersection Scores")
fig = ax.get_figure()
fig.savefig("Bull_Hist_Intersect_Scores.png")
```

| Boston Bull & German Shepherd Histogram  |
|---|
|![GerBull_Hists](/Images/GerBull_Hists.png)|

Boston Bull Intersection Scores           |  German Shepherd Intersection Scores
:-------------------------:|:-------------------------:
![Bull_Hist_Intersect_Scores](/Images/Bull_Hist_Intersect_Scores.png)  |  ![GerShep_Hist_Intersect_Scores](/Images/GerShep_Hist_Intersect_Scores.png)

### 2)
