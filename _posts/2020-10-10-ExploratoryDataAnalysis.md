---
title: "Data wrangling shortcuts"
date: 2020-10-10
tags: [Data Wrangling, Exploratory Data Analysis, shortcut]
excerpt: "Predicting collision severity using classifiers"
mathjax: "true"
categories: datawrangling
toc: true
toc_label: "Table of contents"
toc_icon: "cog"
---
by Aliah H.

Here is a compilation of a few shortcuts for data wrangling using Python.

Abbreviations used here for your reference:
* pd : Pandas library
* df : your dataframe

## Checking for missing values


```python
# lists all columns with null values in descending order
missing_val=pd.DataFrame({'Null':df.isnull().sum()})
missing_val.sort_values(by='Null', ascending=False)
```

## Data Encoding

### Replace values

1. Multiple column replacing


```python
# Great when you have multiple columns with common value that needs replacing
cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df[cols] = df[cols].replace({'No internet service':'No'})
```



2. Single column replacing



```python
df['UNDERINFL'] = df['UNDERINFL'].replace(['N','Y'],['0','1'])
```

### Binary encoding?? for transforming categorical data into numerical 


```python
# Great when you have not a lot of categorical attributes to transform

# le_attname= le_attribute name
from sklearn import preprocessing

# Example for address type attribute
le_addrtype = preprocessing.LabelEncoder()
le_addrtype.fit(['Block', 'Intersection', 'Alley'])
X[:,2] = le_addrtype.transform(X[:,2]) 

le_colltype=preprocessing.LabelEncoder()
le_colltype.fit(['Rear Ended','Angles','Parked Car','Other','Sideswipe','Left Turn','Pedestrian','Cycles','Right Turn','Head On'])
X[:,3]= le_colltype.transform(X[:,3])
```


```python
# First identify the columns that are categorical (dtype=object)
df2 = df.select_dtypes(include=['object']).copy() #copied the original df too to make sure no changes was made directly

# Then extract the categorical column names into list
objList = df2.select_dtypes(include = "object").columns # this is extracted from the original df
print (objList)

#then convert the categorical values into numerical ones
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    df2[feat] = le.fit_transform(df2[feat].astype(str))

print (df2.info())
```

## To reverse transform 


```python
encoded_data, mapping_index = pd.Series(df['MultipleLines'].tolist()).factorize()
```


```python
print(encoded_data)
print(mapping_index)
print(mapping_index.get_loc("No phone service"))
```

## Balancing the dataset

### Upsampling minority class


```python
# First check for nan values in the target label, if there is then drop the corresponding entry
check_for_nan=df['SEVERITYCODE'].isnull().values.any()
print(check_for_nan)

# Then we count the number of each unique entry in the target label 

df['SEVERITYCODE'].value_counts().to_frame()

# as an example, this returned 1:130634   2:56870

# Then, we can upsample the minority class 
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.SEVERITYCODE==1]
df_minority = df[df.SEVERITYCODE==2]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=130634,    # to match majority class in df2 (NaN removed)
                                 random_state=4) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.SEVERITYCODE.value_counts()

#Then we can use this new dataframe (df_upsampled) for model training

```

## To list all values from a column


```python
cat_list= df['MultipleLines'].tolist()
cat_list
```

**And the list goes on! I will add more as I go along with my projects. Stay tuned!**
