# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

<img width="642" height="308" alt="image" src="https://github.com/user-attachments/assets/fc592d6d-7a31-4ed0-b078-2e3981dec9bc" />


```
df.dropna()
```

<img width="772" height="570" alt="image" src="https://github.com/user-attachments/assets/501b68f8-5fc5-4689-8501-7ce9ff554c8c" />


```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

<img width="725" height="110" alt="image" src="https://github.com/user-attachments/assets/caf6bc42-173b-4aba-8c26-749b13447741" />


```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="691" height="438" alt="image" src="https://github.com/user-attachments/assets/45caf6df-f236-4963-b0fa-9fb4175a3b74" />


```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="866" height="525" alt="image" src="https://github.com/user-attachments/assets/7bfcfaff-513a-4e4d-b11f-4a9d9fdd0a92" />


```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="703" height="560" alt="image" src="https://github.com/user-attachments/assets/af93b3fb-24bc-4950-bac3-34ddf4a10f34" />


```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

<img width="771" height="623" alt="image" src="https://github.com/user-attachments/assets/d3419e3e-b68a-4c74-b5cd-77f9b917c9f2" />

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

<img width="551" height="246" alt="image" src="https://github.com/user-attachments/assets/9a4d02e6-0c0f-4b12-b263-c7824a41afc3" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="771" height="250" alt="image" src="https://github.com/user-attachments/assets/febfb1c7-9411-43a1-9a44-bfbf337c66bb" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

<img width="661" height="187" alt="image" src="https://github.com/user-attachments/assets/de645e97-e64e-4d50-9a0d-ccc723749be4" />

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print("Chi-squared statistic:", chi2)
print("p-value:", p)
```

<img width="790" height="152" alt="image" src="https://github.com/user-attachments/assets/ea0f8a11-e771-4829-8c38-9dd6114c1f85" />


```
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']

selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="962" height="477" alt="image" src="https://github.com/user-attachments/assets/810eadcf-2e03-4862-94e5-92d46e06cdd6" />

# RESULT:
Thus the python code to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file is implemented.

