---
title: "Building breast cancer diagnostic classifier with KNN"
date: 2020-07-13
tags: [KNN, Classifier, Machine Learning]
excerpt: "KNN, CLassifier, Machine Learning"
mathjax: "true"
categories: CATEGORY-1 CATEGORY-2
---
by Aliah H.


<a img src = "/images/KNN_images/pink-ribbon.png " width = 400, align = "center"></a>

<h1 align=center><font size = 5> Is the cancer benign or malignant?</font></h1>

```python
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
```

## Let's have a look at the data 

I downloaded the data from Kaggle [https://www.kaggle.com/uciml/breast-cancer-wisconsin-data]


```python
# Load csv data, data has a header
df= pd.read_csv('Breast_cancer_data.csv') 

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
df.shape
```




    (569, 33)



Attribute Information:

1) ID number 

2) Diagnosis (M = malignant, B = benign) 
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) 

b) texture (standard deviation of gray-scale values)

c) perimeter 

d) area 

e) smoothness (local variation in radius lengths)  

f) compactness (perimeter^2 / area - 1.0) 

g) concavity (severity of concave portions of the contour) 

h) concave points (number of concave portions of the contour) 

i) symmetry 

j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

### We don't need the last column since it is not contributing any information


```python
df=df.drop('Unnamed: 32', axis=1)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



## Define feature set, X


```python
df.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')




```python
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']].values

y=df['diagnosis'].values
```

## Normalize X


```python
from sklearn.preprocessing import StandardScaler

X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
```

## Split data into training and testing sets 


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```

    Train set: (455, 30) (455,)
    Test set: (114, 30) (114,)


## Find the best number of neighbours, k


```python
from sklearn.neighbors import KNeighborsClassifier #to build model
from sklearn import metrics #to test accuracy

Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    KNNmodel = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=KNNmodel.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

#Visualize the Ks   
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
```


![png](/images/KNN_images/BC_19_0.png)


    The best accuracy was with 0.9649122807017544 with k= 3


## Build the classifier and make predictions


```python
# starting with k=3 as recommended above 
classifier = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)  # the model

y_pred = classifier.predict(X_test) #predict y (the diagnosis)

w = metrics.accuracy_score(y_test, y_pred) # get accuracy score

print("accuracy score:", w)
```

    accuracy score: 0.9649122807017544



```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

cf_matrix=confusion_matrix(y_test, y_pred)

cat = ["malignant", "benign"]
sns.heatmap(cf_matrix, annot=True, xticklabels=cat, yticklabels=cat, cmap='Blues')

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               B       0.95      1.00      0.97        75
               M       1.00      0.90      0.95        39
    
        accuracy                           0.96       114
       macro avg       0.97      0.95      0.96       114
    weighted avg       0.97      0.96      0.96       114
    



![png](/images/KNN_images/BC_22_1.png)


The confusion matrix shows that the model was able to predict malignancy with 100% accuracy but lesser accuracy (but still good - 95%) for benign. 


```python
# Create dataframes from Xtest, ytest and ypred arrays
df_ypred=pd.DataFrame(data=y_pred)
df_Xtest=pd.DataFrame(data=X_test)
df_ytest=pd.DataFrame(data=y_test)

# Name the columns
df_Xtest.columns=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
df_ytest.columns=["Diagnosis"]
df_ypred.columns=["Predicted"]

# Concatenate the dataframes so that we can easily compare predictions with the test set
frames=(df_ypred, df_ytest,  df_Xtest)

result=pd.concat(frames, axis=1)

result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>M</td>
      <td>-0.368449</td>
      <td>0.707510</td>
      <td>-0.276346</td>
      <td>-0.431419</td>
      <td>0.885278</td>
      <td>1.431955</td>
      <td>1.013195</td>
      <td>0.507906</td>
      <td>...</td>
      <td>-0.221411</td>
      <td>0.728364</td>
      <td>-0.058416</td>
      <td>-0.306902</td>
      <td>1.987143</td>
      <td>1.781414</td>
      <td>1.707974</td>
      <td>1.265235</td>
      <td>0.818992</td>
      <td>2.236260</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>M</td>
      <td>0.600039</td>
      <td>-0.120926</td>
      <td>0.693271</td>
      <td>0.427215</td>
      <td>0.728714</td>
      <td>1.437641</td>
      <td>1.330836</td>
      <td>1.073052</td>
      <td>...</td>
      <td>0.472316</td>
      <td>-0.095626</td>
      <td>0.584958</td>
      <td>0.264420</td>
      <td>0.181104</td>
      <td>1.376193</td>
      <td>1.105405</td>
      <td>0.892184</td>
      <td>-0.211534</td>
      <td>1.238775</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>M</td>
      <td>1.764497</td>
      <td>0.516691</td>
      <td>1.809525</td>
      <td>1.732374</td>
      <td>1.468835</td>
      <td>1.575986</td>
      <td>2.105477</td>
      <td>2.617595</td>
      <td>...</td>
      <td>1.870123</td>
      <td>1.006827</td>
      <td>1.901492</td>
      <td>1.858847</td>
      <td>1.176179</td>
      <td>1.240059</td>
      <td>1.257966</td>
      <td>2.343279</td>
      <td>4.298838</td>
      <td>1.022654</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>B</td>
      <td>-0.078755</td>
      <td>0.072221</td>
      <td>-0.135476</td>
      <td>-0.177157</td>
      <td>-0.677515</td>
      <td>-0.777787</td>
      <td>-0.946385</td>
      <td>-0.670364</td>
      <td>...</td>
      <td>-0.132365</td>
      <td>0.379878</td>
      <td>-0.189474</td>
      <td>-0.231136</td>
      <td>-0.901643</td>
      <td>-0.891646</td>
      <td>-1.077804</td>
      <td>-0.848216</td>
      <td>-0.627304</td>
      <td>-0.822139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>M</td>
      <td>1.114105</td>
      <td>-0.730617</td>
      <td>1.162839</td>
      <td>0.998595</td>
      <td>0.721598</td>
      <td>2.089571</td>
      <td>0.999384</td>
      <td>1.523931</td>
      <td>...</td>
      <td>1.267513</td>
      <td>-1.102000</td>
      <td>1.275989</td>
      <td>1.282251</td>
      <td>0.676449</td>
      <td>1.966530</td>
      <td>0.510512</td>
      <td>1.455568</td>
      <td>1.375509</td>
      <td>1.488146</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>B</td>
      <td>B</td>
      <td>-2.029648</td>
      <td>-1.363580</td>
      <td>-1.984504</td>
      <td>-1.454443</td>
      <td>1.468835</td>
      <td>-0.543168</td>
      <td>-1.114873</td>
      <td>-1.261820</td>
      <td>...</td>
      <td>-1.726901</td>
      <td>-0.999409</td>
      <td>-1.693361</td>
      <td>-1.222423</td>
      <td>1.141110</td>
      <td>-0.852841</td>
      <td>-1.305831</td>
      <td>-1.745063</td>
      <td>0.050546</td>
      <td>0.547186</td>
    </tr>
    <tr>
      <th>110</th>
      <td>M</td>
      <td>M</td>
      <td>0.270583</td>
      <td>1.501040</td>
      <td>0.248417</td>
      <td>0.175512</td>
      <td>0.429819</td>
      <td>-0.126046</td>
      <td>0.435666</td>
      <td>0.428460</td>
      <td>...</td>
      <td>0.464033</td>
      <td>1.228294</td>
      <td>0.415178</td>
      <td>0.297820</td>
      <td>1.474263</td>
      <td>-0.118736</td>
      <td>0.627092</td>
      <td>0.578516</td>
      <td>-0.399197</td>
      <td>0.578219</td>
    </tr>
    <tr>
      <th>111</th>
      <td>B</td>
      <td>B</td>
      <td>-0.507616</td>
      <td>-1.633519</td>
      <td>-0.536668</td>
      <td>-0.530110</td>
      <td>-0.450497</td>
      <td>-0.782146</td>
      <td>-0.743497</td>
      <td>-0.579053</td>
      <td>...</td>
      <td>-0.550672</td>
      <td>-1.043377</td>
      <td>-0.596944</td>
      <td>-0.554943</td>
      <td>-0.138898</td>
      <td>-0.298127</td>
      <td>-0.446594</td>
      <td>-0.115817</td>
      <td>0.338511</td>
      <td>-0.444757</td>
    </tr>
    <tr>
      <th>112</th>
      <td>M</td>
      <td>M</td>
      <td>2.238801</td>
      <td>0.607446</td>
      <td>2.274975</td>
      <td>2.352388</td>
      <td>0.707364</td>
      <td>1.725703</td>
      <td>1.958584</td>
      <td>2.609857</td>
      <td>...</td>
      <td>2.358838</td>
      <td>0.019993</td>
      <td>2.613373</td>
      <td>2.366883</td>
      <td>-0.130131</td>
      <td>0.853922</td>
      <td>0.975872</td>
      <td>1.958046</td>
      <td>-0.258450</td>
      <td>0.099426</td>
    </tr>
    <tr>
      <th>113</th>
      <td>M</td>
      <td>M</td>
      <td>0.262062</td>
      <td>-0.051114</td>
      <td>0.217936</td>
      <td>0.133704</td>
      <td>-0.299627</td>
      <td>-0.348157</td>
      <td>-0.175008</td>
      <td>-0.143650</td>
      <td>...</td>
      <td>0.271446</td>
      <td>0.388020</td>
      <td>0.194763</td>
      <td>0.151913</td>
      <td>-0.340543</td>
      <td>-0.280951</td>
      <td>0.069140</td>
      <td>-0.039684</td>
      <td>-1.001011</td>
      <td>-0.798310</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 32 columns</p>
</div>



## Success!


```python

```
