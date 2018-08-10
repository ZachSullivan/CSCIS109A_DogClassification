---
title: Models
---

## Contents
{:.no_toc}
*  
{: toc}

**Model Descriptions**

## 0. Data Preparation
### 1) Reading and Cleaning Data

```python

```

### 2) Custom Functions
```python
# Return a beautiful soup object from the provided file
def obtain_annotations (filename):    
    if os.path.exists(filename):
        with open(filename) as fp:
            return BeautifulSoup(fp, "lxml")

# Return the bounding box x,y dimensions for the given image
def obtain_boundBox_loc (soup):
    # Use beautiful soup to parse the label file, ensure our output variables are integer dtypes
    xmin = int(soup.find("xmin").text)
    xmax = int(soup.find("xmax").text)
    ymin = int(soup.find("ymin").text)
    ymax = int(soup.find("ymax").text)
    
    return (xmin, xmax, ymin, ymax)

# Given the dog label, return a enumerated value that corresponds with the dogs breed (0 = bull, 1 = gershep)
def obtain_dogclass (soup):
    breed = soup.find("name").text
    if (breed == 'Boston_bull'):
        return 0
    elif (breed == 'German_shepherd'):
        return 1
    else:
        return None
```
```python
# Crop the provided image based on the provided bounding dimensions 
def generate_boundBox (img, xmin, xmax, ymin, ymax):
    crop_img = img[ymin:ymax, xmin:xmax]
    return crop_img
```
```python
# Resize our image to a uniform size and flatten it
# Return a new list containing the raw pixel values
def image_to_feature_vector(image, size):
    return cv2.resize(image, (size, size)).flatten()
```
```python
def load_images_from_folder(imageFolder, labelFolder, size):
    pixValues = []
    classes = []
    for filename in os.listdir(imageFolder):
        
        img = cv2.imread(os.path.join(imageFolder, filename))
        
        if img is not None:
            
            # Obtain the label information for that picture
            bsObj = obtain_annotations(labelFolder + filename[:-4])
            
            # Obtain the dimensions for the location of the dog
            xmin, xmax, ymin, ymax = obtain_boundBox_loc(bsObj)
            
            # Crop our image such that only the dog is shown
            crop_img = generate_boundBox(img, xmin, xmax, ymin, ymax)
            
            # Obtain a 1D array of every pixel value in the image
            pixValues.append(image_to_feature_vector(crop_img, size))
            
            # Obtain the corresponding enumerated classification value for the dog's breed
            classes.append(obtain_dogclass(bsObj))

    return np.array(pixValues), np.array(classes)
```
```python
# Appends list values to selected columns given a provided dataframe 
# Returns the update dataframe
def DfAppend_Vals(dataFrame, matrix, columns):
    for vector in matrix:
        dataFrame = dataFrame.append(pd.Series(vector, index = columns), ignore_index=True)

    return dataFrame
```

## 1. Classification Varient 1

### 1) Model 1

### 2) Model 2

### 3) Model 3

### 4) Model 4

## 2. Classification Varient 1

### 1) Model 1

### 2) Model 2

### 3) Model 3

### 4) Model 4
