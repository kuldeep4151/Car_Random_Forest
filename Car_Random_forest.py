#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv('/Users/kuldeeppatel/Downloads/car_evaluation.csv')

# %%
df.head()
# %%
df.info()
df = df.rename(columns={
    "vhigh": "buying",
    "vhigh.1": "maint",
    "2": "doors",
    "2.1": "persons",
    "small": "lug_boot",
    "low": "safety",
    "unacc": "decision",
})# %%

# %%
df.info()
df.shape
# %%
#Check frequency count
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'decision']


for col in col_names:
    
    print(df[col].value_counts())
# %%
df.isnull().sum()

# %%
X = df.drop(['decision'], axis=1)

y = df['decision']


# %%
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
df.shape
# %%
X_train.shape, X_test.shape
# %%
import category_encoders as ce
# encode categorical variables with ordinal encoding

# %%
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
# %%
X_train.head()
# %%

model = RandomForestClassifier()
model.fit(X_train, y_train)


# %%
y_pred = model.predict(X_test)
# %%
from sklearn.metrics import accuracy_score

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# %%
