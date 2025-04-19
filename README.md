# Feature-Scaling-and-Selection-Completion-requirements

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

df = pd.read_csv('/content/bmi.csv')

df.head()

df.dropna()

max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals

df1 = pd.read_csv('/content/bmi.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1[['Height','Weight']] = sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df.head(10)

df2 = pd.read_csv('/content/bmi.csv')

from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df2[['Height','Weight']] = scaler.fit_transform(df2[['Height','Weight']])
df2

df3 = pd.read_csv('/content/bmi.csv')

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height','Weight']] = scaler.fit_transform(df3[['Height','Weight']])
df3

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4 = pd.read_csv('/content/bmi.csv')
df4[['Height','Weight']] = scaler.fit_transform(df4[['Height','Weight']])
df4.head()

from scipy.stats import chi2_contingency
import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()

contigency_table = pd.crosstab(tips['sex'],tips['time'])
contigency_table

chi2 , p , _ , _ , = chi2_contingency(contigency_table)
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")

from sklearn.feature_selection import SelectKBest, mutual_info_classif
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B' , 'C' , 'A' , 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['Feature1', 'Feature3']]
y = df['Target']
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)
selected_features_indices = selector.get_support(indices = True)
selected_features = X.columns[selected_features_indices]
print("Selected Features")
print(selected_features)
```
# RESULT:
       # INCLUDE YOUR RESULT HERE
