---
title: "COVID-19 analysis with Python"
date: 2020-10-13
tags: [Data Wrangling, COVID-19, regplot]
excerpt: "Analysing the relationship between infection and well being of countries"
mathjax: "true"
categories: datawrangling
toc: true
toc_label: "Table of contents"
toc_icon: "cog"
---
# COVID-19 analysis with Python
by Aliah H.

In this project, I used Python to analyse three datasets:- 1) Global confirmed cases of COVID-19 infection, 2) Global confirmed cases of COVID-19 related deaths and 3) Worldwide happiness report 2020. 

I explored these datasets to find the relationship between the infection rates or death rates with the conditions in each country. 

## The datasets: 

COVID-19 global dataset: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases?force_layout=desktop 

COVID-19 global death dataset: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases?force_layout=desktop

World happiness report 2020: https://www.kaggle.com/mathurinache/world-happiness-report


```python
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
```

## Global confirmed COVID-19 cases dataset


```python
covid19_df=pd.read_csv("time_series_covid19_confirmed_global.csv")

covid19_df.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>39290</td>
      <td>39297</td>
      <td>39341</td>
      <td>39422</td>
      <td>39486</td>
      <td>39548</td>
      <td>39616</td>
      <td>39693</td>
      <td>39703</td>
      <td>39799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13965</td>
      <td>14117</td>
      <td>14266</td>
      <td>14410</td>
      <td>14568</td>
      <td>14730</td>
      <td>14899</td>
      <td>15066</td>
      <td>15231</td>
      <td>15399</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51847</td>
      <td>51995</td>
      <td>52136</td>
      <td>52270</td>
      <td>52399</td>
      <td>52520</td>
      <td>52658</td>
      <td>52804</td>
      <td>52940</td>
      <td>53072</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2110</td>
      <td>2110</td>
      <td>2110</td>
      <td>2370</td>
      <td>2370</td>
      <td>2568</td>
      <td>2568</td>
      <td>2696</td>
      <td>2696</td>
      <td>2696</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5211</td>
      <td>5370</td>
      <td>5402</td>
      <td>5530</td>
      <td>5725</td>
      <td>5725</td>
      <td>5958</td>
      <td>6031</td>
      <td>6246</td>
      <td>6366</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 268 columns</p>
</div>




```python
print(covid19_df.shape)
```

    (266, 268)



```python
covid19_df.drop(['Lat', 'Long'], axis=1, inplace=True)

covid19_df.head(10)
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>39290</td>
      <td>39297</td>
      <td>39341</td>
      <td>39422</td>
      <td>39486</td>
      <td>39548</td>
      <td>39616</td>
      <td>39693</td>
      <td>39703</td>
      <td>39799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13965</td>
      <td>14117</td>
      <td>14266</td>
      <td>14410</td>
      <td>14568</td>
      <td>14730</td>
      <td>14899</td>
      <td>15066</td>
      <td>15231</td>
      <td>15399</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51847</td>
      <td>51995</td>
      <td>52136</td>
      <td>52270</td>
      <td>52399</td>
      <td>52520</td>
      <td>52658</td>
      <td>52804</td>
      <td>52940</td>
      <td>53072</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2110</td>
      <td>2110</td>
      <td>2110</td>
      <td>2370</td>
      <td>2370</td>
      <td>2568</td>
      <td>2568</td>
      <td>2696</td>
      <td>2696</td>
      <td>2696</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5211</td>
      <td>5370</td>
      <td>5402</td>
      <td>5530</td>
      <td>5725</td>
      <td>5725</td>
      <td>5958</td>
      <td>6031</td>
      <td>6246</td>
      <td>6366</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Antigua and Barbuda</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>106</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>108</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>Argentina</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>779689</td>
      <td>790818</td>
      <td>798486</td>
      <td>809728</td>
      <td>824468</td>
      <td>840915</td>
      <td>856369</td>
      <td>871468</td>
      <td>883882</td>
      <td>894206</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Armenia</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51382</td>
      <td>51925</td>
      <td>52496</td>
      <td>52677</td>
      <td>53083</td>
      <td>53755</td>
      <td>54473</td>
      <td>55087</td>
      <td>55736</td>
      <td>56451</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Australian Capital Territory</td>
      <td>Australia</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
      <td>113</td>
    </tr>
    <tr>
      <th>9</th>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>4232</td>
      <td>4234</td>
      <td>4235</td>
      <td>4246</td>
      <td>4249</td>
      <td>4261</td>
      <td>4271</td>
      <td>4273</td>
      <td>4278</td>
      <td>4284</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 266 columns</p>
</div>




```python
covid19_agg=covid19_df.groupby("Country/Region").sum()
covid19_agg.head(10)
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>39290</td>
      <td>39297</td>
      <td>39341</td>
      <td>39422</td>
      <td>39486</td>
      <td>39548</td>
      <td>39616</td>
      <td>39693</td>
      <td>39703</td>
      <td>39799</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13965</td>
      <td>14117</td>
      <td>14266</td>
      <td>14410</td>
      <td>14568</td>
      <td>14730</td>
      <td>14899</td>
      <td>15066</td>
      <td>15231</td>
      <td>15399</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51847</td>
      <td>51995</td>
      <td>52136</td>
      <td>52270</td>
      <td>52399</td>
      <td>52520</td>
      <td>52658</td>
      <td>52804</td>
      <td>52940</td>
      <td>53072</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2110</td>
      <td>2110</td>
      <td>2110</td>
      <td>2370</td>
      <td>2370</td>
      <td>2568</td>
      <td>2568</td>
      <td>2696</td>
      <td>2696</td>
      <td>2696</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5211</td>
      <td>5370</td>
      <td>5402</td>
      <td>5530</td>
      <td>5725</td>
      <td>5725</td>
      <td>5958</td>
      <td>6031</td>
      <td>6246</td>
      <td>6366</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>106</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>108</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>779689</td>
      <td>790818</td>
      <td>798486</td>
      <td>809728</td>
      <td>824468</td>
      <td>840915</td>
      <td>856369</td>
      <td>871468</td>
      <td>883882</td>
      <td>894206</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51382</td>
      <td>51925</td>
      <td>52496</td>
      <td>52677</td>
      <td>53083</td>
      <td>53755</td>
      <td>54473</td>
      <td>55087</td>
      <td>55736</td>
      <td>56451</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>9</td>
      <td>9</td>
      <td>...</td>
      <td>27121</td>
      <td>27135</td>
      <td>27148</td>
      <td>27173</td>
      <td>27181</td>
      <td>27206</td>
      <td>27226</td>
      <td>27244</td>
      <td>27263</td>
      <td>27285</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>46374</td>
      <td>47432</td>
      <td>48146</td>
      <td>48896</td>
      <td>49819</td>
      <td>50848</td>
      <td>52057</td>
      <td>53188</td>
      <td>54423</td>
      <td>55319</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 264 columns</p>
</div>




```python
covid19_agg.loc['Malaysia'].plot()
covid19_agg.loc['Singapore'].plot()
covid19_agg.loc['China'].plot()

plt.title("Number of cases since January 2020", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a245aae10>




![png](/images/DW/COVID/output_10_1.png)


## Maximum rate of infection 

Calculating the maximum rate of infection for each country


```python
countries = list(covid19_agg.index)

max_infection_rates = []

for country in countries :
    max_infection_rates.append(covid19_agg.loc[country].diff().max())

# Adding the max infection rate column to the dataframe
covid19_agg['max_infection_rate'] = max_infection_rates
```


```python
covid19_agg.head(5)
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
      <th>max_infection_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>39297</td>
      <td>39341</td>
      <td>39422</td>
      <td>39486</td>
      <td>39548</td>
      <td>39616</td>
      <td>39693</td>
      <td>39703</td>
      <td>39799</td>
      <td>915.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14117</td>
      <td>14266</td>
      <td>14410</td>
      <td>14568</td>
      <td>14730</td>
      <td>14899</td>
      <td>15066</td>
      <td>15231</td>
      <td>15399</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>51995</td>
      <td>52136</td>
      <td>52270</td>
      <td>52399</td>
      <td>52520</td>
      <td>52658</td>
      <td>52804</td>
      <td>52940</td>
      <td>53072</td>
      <td>675.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2110</td>
      <td>2110</td>
      <td>2370</td>
      <td>2370</td>
      <td>2568</td>
      <td>2568</td>
      <td>2696</td>
      <td>2696</td>
      <td>2696</td>
      <td>260.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5370</td>
      <td>5402</td>
      <td>5530</td>
      <td>5725</td>
      <td>5725</td>
      <td>5958</td>
      <td>6031</td>
      <td>6246</td>
      <td>6366</td>
      <td>233.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 265 columns</p>
</div>




```python
covid_data=pd.DataFrame(covid19_agg['max_infection_rate'])
covid_data.head()
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
      <th>max_infection_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>915.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>178.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>675.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>260.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>233.0</td>
    </tr>
  </tbody>
</table>
</div>



## World happiness report 2020 dataset


```python
happiness_report=pd.read_csv("world_happiness_report_2020.csv")

happiness_report.head()
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
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>0.031156</td>
      <td>7.869766</td>
      <td>7.747634</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>1.285190</td>
      <td>1.499526</td>
      <td>0.961271</td>
      <td>0.662317</td>
      <td>0.159670</td>
      <td>0.477857</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>0.033492</td>
      <td>7.711245</td>
      <td>7.579955</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>1.326949</td>
      <td>1.503449</td>
      <td>0.979333</td>
      <td>0.665040</td>
      <td>0.242793</td>
      <td>0.495260</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>0.035014</td>
      <td>7.628528</td>
      <td>7.491272</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>1.390774</td>
      <td>1.472403</td>
      <td>1.040533</td>
      <td>0.628954</td>
      <td>0.269056</td>
      <td>0.407946</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>0.059616</td>
      <td>7.621347</td>
      <td>7.387653</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>1.326502</td>
      <td>1.547567</td>
      <td>1.000843</td>
      <td>0.661981</td>
      <td>0.362330</td>
      <td>0.144541</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>0.034837</td>
      <td>7.556281</td>
      <td>7.419719</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>1.424207</td>
      <td>1.495173</td>
      <td>1.008072</td>
      <td>0.670201</td>
      <td>0.287985</td>
      <td>0.434101</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>
</div>




```python
happiness_report.shape
```




    (153, 20)




```python
drop_cols=['Ladder score', 'Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Generosity',
          'Perceptions of corruption', 'Explained by: Log GDP per capita', 'Explained by: Social support',
          'Explained by: Healthy life expectancy','Explained by: Freedom to make life choices', 'Explained by: Generosity',
          'Explained by: Perceptions of corruption','Dystopia + residual', 'Regional indicator','Ladder score in Dystopia']

happiness_report.drop(drop_cols, axis=1, inplace=True)
happiness_report.head()
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
      <th>Country name</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Switzerland</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iceland</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Norway</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
    </tr>
  </tbody>
</table>
</div>




```python
happiness_report.set_index(['Country name'],inplace=True)
happiness_report.head()
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
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
    <tr>
      <th>Country name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finland</th>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
    </tr>
  </tbody>
</table>
</div>



# Joining both datasets

### COVID-19 data


```python
covid_data.shape
```




    (188, 1)




```python
happiness_report.shape
```




    (153, 4)




```python
data = covid_data.join(happiness_report, how='inner')
data.head()
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
      <th>max_infection_rate</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>915.0</td>
      <td>7.462861</td>
      <td>0.470367</td>
      <td>52.590000</td>
      <td>0.396573</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>178.0</td>
      <td>9.417931</td>
      <td>0.671070</td>
      <td>68.708138</td>
      <td>0.781994</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>675.0</td>
      <td>9.537965</td>
      <td>0.803385</td>
      <td>65.905174</td>
      <td>0.466611</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>16447.0</td>
      <td>9.810955</td>
      <td>0.900568</td>
      <td>68.803802</td>
      <td>0.831132</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>771.0</td>
      <td>9.100476</td>
      <td>0.757479</td>
      <td>66.750656</td>
      <td>0.712018</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory data analysis


```python
corr=data.corr()
corr
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
      <th>max_infection_rate</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>max_infection_rate</th>
      <td>1.000000</td>
      <td>0.123782</td>
      <td>0.054207</td>
      <td>0.123502</td>
      <td>0.101480</td>
    </tr>
    <tr>
      <th>Logged GDP per capita</th>
      <td>0.123782</td>
      <td>1.000000</td>
      <td>0.788877</td>
      <td>0.858725</td>
      <td>0.440761</td>
    </tr>
    <tr>
      <th>Social support</th>
      <td>0.054207</td>
      <td>0.788877</td>
      <td>1.000000</td>
      <td>0.764977</td>
      <td>0.486838</td>
    </tr>
    <tr>
      <th>Healthy life expectancy</th>
      <td>0.123502</td>
      <td>0.858725</td>
      <td>0.764977</td>
      <td>1.000000</td>
      <td>0.478732</td>
    </tr>
    <tr>
      <th>Freedom to make life choices</th>
      <td>0.101480</td>
      <td>0.440761</td>
      <td>0.486838</td>
      <td>0.478732</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```


![png](/images/DW/COVID/output_27_0.png)



```python
x=data['Logged GDP per capita']
y=data['max_infection_rate']

plt.title("GDP per capita vs. maximum infection rate", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')

sns.regplot(x,np.log(y))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a24712f50>




![png](/images/DW/COVID/output_28_1.png)



```python
x=data['Social support']
y=data['max_infection_rate']

plt.title("Social support vs. maximum infection rate", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')

sns.regplot(x,np.log(y))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2572f8d0>




![png](/images/DW/COVID/output_29_1.png)



```python
x=data['Healthy life expectancy']
y=data['max_infection_rate']

plt.title("Healthy life expectancy vs. maximum infection rate", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')

sns.regplot(x,np.log(y))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2572f310>




![png](/images/DW/COVID/output_30_1.png)



```python
x=data['Freedom to make life choices']
y=data['max_infection_rate']

plt.title("Freedom to make life choices vs. maximum infection rate", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')

sns.regplot(x,np.log(y))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25abdfd0>




![png](/images/DW/COVID/output_31_1.png)



```python
x=data['Social support']
y=data['Logged GDP per capita']

plt.title("Social support vs. GDP per capita", 
          fontdict={'fontsize': 14,
                    'fontweight' : 'bold',
                    'verticalalignment': 'baseline'}, loc='center')

sns.regplot(x,np.log(y))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25b9e850>




![png](/images/DW/COVID/output_32_1.png)


## Global COVID-19 related death dataset


```python
covid19_death=pd.read_csv("time_series_covid19_deaths_global.csv")
covid19_death.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1458</td>
      <td>1462</td>
      <td>1462</td>
      <td>1466</td>
      <td>1467</td>
      <td>1469</td>
      <td>1470</td>
      <td>1472</td>
      <td>1473</td>
      <td>1477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>389</td>
      <td>392</td>
      <td>396</td>
      <td>400</td>
      <td>403</td>
      <td>407</td>
      <td>411</td>
      <td>413</td>
      <td>416</td>
      <td>420</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1749</td>
      <td>1756</td>
      <td>1760</td>
      <td>1768</td>
      <td>1768</td>
      <td>1771</td>
      <td>1783</td>
      <td>1789</td>
      <td>1795</td>
      <td>1801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>54</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>189</td>
      <td>193</td>
      <td>195</td>
      <td>199</td>
      <td>211</td>
      <td>211</td>
      <td>208</td>
      <td>212</td>
      <td>218</td>
      <td>218</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 268 columns</p>
</div>




```python
covid19_death.shape
```




    (266, 268)




```python
covid19_death.drop(['Lat', 'Long'], axis=1, inplace=True)
covid19_death.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1458</td>
      <td>1462</td>
      <td>1462</td>
      <td>1466</td>
      <td>1467</td>
      <td>1469</td>
      <td>1470</td>
      <td>1472</td>
      <td>1473</td>
      <td>1477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>389</td>
      <td>392</td>
      <td>396</td>
      <td>400</td>
      <td>403</td>
      <td>407</td>
      <td>411</td>
      <td>413</td>
      <td>416</td>
      <td>420</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1749</td>
      <td>1756</td>
      <td>1760</td>
      <td>1768</td>
      <td>1768</td>
      <td>1771</td>
      <td>1783</td>
      <td>1789</td>
      <td>1795</td>
      <td>1801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>54</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>189</td>
      <td>193</td>
      <td>195</td>
      <td>199</td>
      <td>211</td>
      <td>211</td>
      <td>208</td>
      <td>212</td>
      <td>218</td>
      <td>218</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 266 columns</p>
</div>




```python
death_agg=covid19_death.groupby("Country/Region").sum()
death_agg.head()
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1458</td>
      <td>1462</td>
      <td>1462</td>
      <td>1466</td>
      <td>1467</td>
      <td>1469</td>
      <td>1470</td>
      <td>1472</td>
      <td>1473</td>
      <td>1477</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>389</td>
      <td>392</td>
      <td>396</td>
      <td>400</td>
      <td>403</td>
      <td>407</td>
      <td>411</td>
      <td>413</td>
      <td>416</td>
      <td>420</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1749</td>
      <td>1756</td>
      <td>1760</td>
      <td>1768</td>
      <td>1768</td>
      <td>1771</td>
      <td>1783</td>
      <td>1789</td>
      <td>1795</td>
      <td>1801</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>54</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>189</td>
      <td>193</td>
      <td>195</td>
      <td>199</td>
      <td>211</td>
      <td>211</td>
      <td>208</td>
      <td>212</td>
      <td>218</td>
      <td>218</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 264 columns</p>
</div>




```python
countries = list(death_agg.index)

max_death_rates = []

for country in countries :
    max_death_rates.append(death_agg.loc[country].diff().max())

# Adding the max infection rate column to the dataframe
death_agg['max_death_rate'] = max_death_rates
```


```python
death_agg.head()
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
      <th>max_death_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1462</td>
      <td>1462</td>
      <td>1466</td>
      <td>1467</td>
      <td>1469</td>
      <td>1470</td>
      <td>1472</td>
      <td>1473</td>
      <td>1477</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>392</td>
      <td>396</td>
      <td>400</td>
      <td>403</td>
      <td>407</td>
      <td>411</td>
      <td>413</td>
      <td>416</td>
      <td>420</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1756</td>
      <td>1760</td>
      <td>1768</td>
      <td>1768</td>
      <td>1771</td>
      <td>1783</td>
      <td>1789</td>
      <td>1795</td>
      <td>1801</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>54</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>193</td>
      <td>195</td>
      <td>199</td>
      <td>211</td>
      <td>211</td>
      <td>208</td>
      <td>212</td>
      <td>218</td>
      <td>218</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 265 columns</p>
</div>




```python
death_data=pd.DataFrame(death_agg["max_death_rate"])
death_data.head()
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
      <th>max_death_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>46.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>30.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
death_data.shape

```




    (188, 1)




```python
data.shape
```




    (141, 5)




```python
data2 = death_data.join(data, how='inner')
data2.head()
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
      <th>max_death_rate</th>
      <th>max_infection_rate</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>46.0</td>
      <td>915.0</td>
      <td>7.462861</td>
      <td>0.470367</td>
      <td>52.590000</td>
      <td>0.396573</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>6.0</td>
      <td>178.0</td>
      <td>9.417931</td>
      <td>0.671070</td>
      <td>68.708138</td>
      <td>0.781994</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>30.0</td>
      <td>675.0</td>
      <td>9.537965</td>
      <td>0.803385</td>
      <td>65.905174</td>
      <td>0.466611</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>3351.0</td>
      <td>16447.0</td>
      <td>9.810955</td>
      <td>0.900568</td>
      <td>68.803802</td>
      <td>0.831132</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>19.0</td>
      <td>771.0</td>
      <td>9.100476</td>
      <td>0.757479</td>
      <td>66.750656</td>
      <td>0.712018</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr2=data2.corr()
```


```python

ax = sns.heatmap(
    corr2, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```


![png](/images/DW/COVID/output_45_0.png)



```python
x=np.log(data2["max_infection_rate"])
y=np.log(data2["max_death_rate"])

sns.regplot(x,y)
```

    /opt/anaconda3/lib/python3.7/site-packages/pandas/core/series.py:679: RuntimeWarning: divide by zero encountered in log
      result = getattr(ufunc, method)(*inputs, **kwargs)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a265c6d10>




![png](/images/DW/COVID/output_46_2.png)


## Results and Discussion

By combining the confirmed COVID-19 cases with the worldwide happiness report, we were able to see the relationship between the maximum infection rates with the general well being of the countries. The well-being of these countries was shown by analysing the relationship between their GDP per capita with their overall social support, healthy life expectancy and freedom to make life choices. The analysis shows that countries with higher social support, healthy life expectancy and freedom to make life choices also are countries with higher GDP per capita. These well developed countries also show higher maximum infection rate. 

When combining the death dataset to the analysis, we were able to then analyse the relationship between maximum infection rate and the death rate in the countries. The results showed a positive correlation. The death rate showed no or low correlation with how well the countries are doing, demonstrating that high infection rates is the only cause for the COVID-19 related deaths. Further showing the seriousness of this pandemic. 

Stay safe everyone!


```python

```
