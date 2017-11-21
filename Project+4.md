
### Project 4 – Predictive Analysis using scikit-learn
#### Nick Ileczko


```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
from sklearn import metrics

mushroomdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',sep=',', header=None, usecols=[0,3,5], names=["Edible","Cap_Color","Odor"])
mushroomdata.replace(to_replace={"Edible":{'e': 0, 'p': 1}}, inplace=True)
mushroomdata.head(10)



```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Edible</th>
      <th>Cap_Color</th>
      <th>Odor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>n</td>
      <td>p</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>y</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>w</td>
      <td>l</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>w</td>
      <td>p</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>g</td>
      <td>n</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>y</td>
      <td>a</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>w</td>
      <td>a</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>w</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>w</td>
      <td>p</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>y</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating dummy var cap_color, odor
Cap_Color = pd.Series(mushroomdata['Cap_Color'])
cc = pd.get_dummies(Cap_Color)
Odor = pd.Series(mushroomdata['Odor'])
od = pd.get_dummies(Odor)

#new combined data frame
mushd = pd.concat([cc, od, mushroomdata['Edible']], axis=1)
cols = list(mushd.iloc[:, :-1])

#setting up training model
X = mushd.iloc[:, :-1].values
Y = mushd.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)


linereg = sklearn.linear_model.LinearRegression()
linereg.fit(X_train, Y_train)
Y_pred = linereg.predict(X_test)
tr = [1, 0]
pr = [1, 0]

```


```python
#Mean of the values in the model for comparison
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
```

    5.93272516767e-16
    


```python
#Removing Odor
X = mushd.iloc[:, 1:10].values
Y = mushd.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)


print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
```

    2.85951394633e-16
    


```python
#Removing Cap_Color
X = mushd.iloc[:, 11:-1].values
Y = mushd.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
```

    0.0588025088502
    

Judging by the data presented, odor is a better indicator of edibility since it is closer to the mean presented perviously. Although it is still better to take into account both Cap_Color and Odor. 
