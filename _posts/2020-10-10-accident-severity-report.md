---
title: "Car accident severity report: A Capstone Project"
date: 2020-10-10
tags: [KNN, Decision Tree, Classifier, Machine Learning]
excerpt: "Predicting collision severity using classifiers"
mathjax: "true"
categories: machinelearning
toc: true
toc_label: "Summary of steps"
toc_icon: "cog"
---
by Aliah H.
 
<h1 align=center><font size = 5> Capstone Project- Car accident severity report</font></h1>

## Applied Data Science Capstone by IBM/Coursera
by Aliah H. 

## Table of Contents
* [Introduction: Business Problem](#introduction)
* [Data](#data)
* [References](#references)

# Introduction / Business Problem

According to the Washington State Department of Transportation 2015 Annual Collision Summary, a crash occured every 4.5 minutes. Devastatingly, a person died in a crash every 16 hours and a person was injured in a crash every 11 minutes . There were a total of 117,053 collisions with 499 of them resulted in fatality and 1752 of them with serious injuries [1]. In 2018 alone, there were 10, 249 police reported collisions on Seattle street [2].

Meanwhile, the US Department of Transportation's National Highway Traffic Safety Administration (NHTSA) motor vehicle crashes imposed USD 836 billion in economic cost and societal harm on the country in 2010. And for the 2017 financial year, NHTSA requests USD 1.181 billion to effectively continue its mission of ensuring safer drivers, safer cars, and safer roads [3].

In this project, we will try to predict the severity of an accident based on road and weather conditions. Specifically, this report will be targeted to stakeholders interested in developing an app or system that could alert and warn drivers about the potential risk they are facing when driving. 


# Data

The dataset used for this project is based on reported car collisions in Seattle, Washington from 2004 to 2020. The dataset lists the severity of each car accidents along with the time and conditions under which each accident occurred. There are 38 attributes with 194673 entries included in the raw dataset. There is also a metadata document provided along with the dataset that describes each attribute.  


# Methodology

**Machine learning method**

In this project, two classifier models were built using Decision Tree and K Nearest Neighbour. Both models are built, analysed and visualised using Python libraries sklearn, pandas, numpy and matplotlib. 

**Exploratory data analysis** 

To get to know the data, the shape, the name and the datatype of the attributes were identified using pandas library. Then, the dataset was checked for missing values and the target label was checked for count balance. 

**Data preparation**

To prepare the dataset for modelling, attributes identified with more than 50% missing values were dropped. Further, only entries with missing values were dropped for the remaining attributes. The minority class for the target label was then upsampled to balance the dataset. 
The identified categorical attributes were then converted into numerical using <html>LabelEncoder</html>. 

**Modelling, Prediction and Analysis**
The balanced dataset was split into training and testing datasets. Both models were built using the train dataset and predictions were carried out using the test dataset. Both models were then analysed for accuracy by calculating the accuracy and F1 score. 



# Result

## Exploratory Data Analysis

For this step, I did some simple exploration to identify missing values and the datatypes of the attributes. The raw data has 38 attributes and 19,4673 entries. There were 6 attributes with more than 50% missing values in the dataset. These attributes were dropped from the dataset. For the remaining attributes, only entries with missing values were ddropped. Majority of the remaining attributes are numerical whereas 7 attributes were found numerical.
A summary is shown below: 

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
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th># of Entries</th>
      <td>194673</td>
    </tr>
    <tr>
      <th># of Attributes</th>
      <td>38</td>
    </tr>
    <tr>
      <th># of columns with &gt;50% missing values</th>
      <td>6</td>
    </tr>
    <tr>
      <th># of Categorical attributes</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



### Dropping columns

There are 38 attributes in the data. In order to choose which features to use for modelling later on, we first identify the columns that would not be useful for the analysis. This is done by identifying the missing values. 

* **PEDROWNOTGRNT** : Whether or not the pedestrian right of way was not granted. (Y/N) 
* **EXCEPTRSNDESC** : No description provided
* **SPEEDING** : Whether or not speeding was a factor in the collision. (Y/N)
* **INATTENTIONIND** : Whether or not collision was due to inattention. (Y/N)
* **INTKEY** : Key that corresponds to the intersection associated with a collision
* **EXCEPTRSNCODE** : No description provided
* **SDOTCOLNUM**: A number given to the collision by SDOT.

These top 6 attributes has more than 50% missing values, therefore would not be useful if included in the data for modelling. Also preliminarily, four more additional attributes identified ('REPORTN0', 'SDOTCOLNUM','X', 'Y') to would not be contributing to the analysis, thus dropped. These attributes were the report ID and the coordinates for the accident. 

Additionally, columns that are not contributing or too complex are also removed. For example, the attribute 'LOCATION' is a categorical data with more than 10,000 different entries and 'SEVERITYDESC' attribute is redundant.

## Data Preparation
### Target label

The target label is 'SEVERITYCODE' and the description is listed in 'SEVERITYDESC'.  

The 'SEVERITYCODE' attribute lists the code that corresponds to the severity of the collision: 
* 3  - fatality
* 2b - serious injury
* 2  - injury
* 1  - prop damage
* 0  - unknown



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
      <th>SEVERITYCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>130634</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56870</td>
    </tr>
  </tbody>
</table>
</div>



The data is not balanced. 


![png](/images/Capstone/output_18_1.png)


### Balancing the dataset by upsampling minority class

The data was balanced by upsampling the minority class 2. This increased the category to 1300634 entries.

![png](/images/Capstone/output_21_1.png)


### Setting up the dataset

Some preprocessing to generate feature set, X. All categorical data from the upsampled dataframe was converted into numerical using LabelEncoder. For 'UNDERINFL' attribute, the values were standardized using a replace method. The dataset was split into training and testing dataset with a 70:30 ratio. 

'UNDERINFL' was categorised into 4 different values that was redundant.



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
      <th>UNDERINFL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N</th>
      <td>138460</td>
    </tr>
    <tr>
      <th>0</th>
      <td>109098</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>7609</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6101</td>
    </tr>
  </tbody>
</table>
</div>



'N' and 'Y' was replaced to '0' and '1' respectively.

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
      <th>UNDERINFL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>247558</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13710</td>
    </tr>
  </tbody>
</table>
</div>



Attributes with categorical datatype is shown in the table below:


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
      <th>ADDRTYPE</th>
      <th>COLLISIONTYPE</th>
      <th>UNDERINFL</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Block</td>
      <td>Sideswipe</td>
      <td>0</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Block</td>
      <td>Parked Car</td>
      <td>0</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Block</td>
      <td>Other</td>
      <td>0</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Intersection</td>
      <td>Angles</td>
      <td>0</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Intersection</td>
      <td>Angles</td>
      <td>0</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INCKEY</th>
    </tr>
    <tr>
      <th>COLDETKEY</th>
    </tr>
    <tr>
      <th>ADDRTYPE</th>
    </tr>
    <tr>
      <th>COLLISIONTYPE</th>
    </tr>
    <tr>
      <th>PERSONCOUNT</th>
    </tr>
    <tr>
      <th>PEDCOUNT</th>
    </tr>
    <tr>
      <th>PEDCYLCOUNT</th>
    </tr>
    <tr>
      <th>VEHCOUNT</th>
    </tr>
    <tr>
      <th>SDOT_COLCODE</th>
    </tr>
    <tr>
      <th>UNDERINFL</th>
    </tr>
    <tr>
      <th>WEATHER</th>
    </tr>
    <tr>
      <th>ROADCOND</th>
    </tr>
    <tr>
      <th>LIGHTCOND</th>
    </tr>
    <tr>
      <th>ST_COLCODE</th>
    </tr>
    <tr>
      <th>SEGLANEKEY</th>
    </tr>
    <tr>
      <th>CROSSWALKKEY</th>
    </tr>
  </tbody>
</table>
</div>



## Decision Tree Classifier

### Modelling

A decision tree was built using the DecisionTreeClassifier function from sklearn's library. The maximum depth for tree was set at 10 and the tree was trained using the training dataset. 

### Prediction

The trained model was then used to predict collision severity using the testing dataset. 

![png](/images/Capstone/output_69_0.png)


### Evaluation

The accuracy score for the tree was 0.73, which was good. Top 3 factors predicting severity is the number of pedestrian, number of cyclists and number of individuals likely to involve in the collision. 

# KNN Classifier

### Modelling

A KNN model was built using the KNeighborsClassifier function from sklearn's library. The accuracy of the model was tested against a range of number of k to identify the best k. The best accuracy was 0.8 with k=1. 

### Evaluation

![png](/images/Capstone/output_71_1.png)


The accuracy score for the best KNN model was 0.80 and the F1 score was 0.78 and 0.82 for predicting 'Property Damage only' and 'Injury' collisions. 

The model is optimised at k=1, at which the model correctly predicts accident severity code 1 and 2- 86% and 76% of the time, respectively. The F1 scores of the two accident outcomes are 0.78 and 0.82.

## Discussion and summary

The scores for the KNN model are higher than of the Decision Tree demonstrating that for this dataset, the KNN model performs better. 

# References 

1. Washington State Department of Transportation 2015 Annual Collision Summary. https://wsdot.wa.gov/mapsdata/crash/pdf/2015_Annual_Collision_Summary.pdf

2. Seattle Department of Transportation 2019 Traffic Report. https://www.seattle.gov/Documents/Departments/SDOT/VisionZero/2019_Traffic_Report.pdf

3. National Highway Traffic Safety Administration (NHTSA) Budget Estimates (Fiscal Year 2017). https://www.nhtsa.gov/sites/nhtsa.dot.gov/files/fy2017-nhtsa_cbj_final_02_2016.pdf

