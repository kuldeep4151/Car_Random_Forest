
#%%
import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()
# %%
dir(digits)
# %%
%matplotlib inline 
import matplotlib.pyplot as plt
plt.gray()
for i in range(7):
    plt.matshow(digits.images[i])


# %%
digits.data[:5]
# %%
df = pd.DataFrame(digits.data)
df.head()
# %%
digits.target
# %%
df['target'] = digits.target
df.head()
# %%
from sklearn.model_selection import train_test_split

X = df.drop(['target'], axis=1,)
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
len(X_train)
# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# %%
model.score(X_test, y_test)
# %%
y_pred = model.predict(X_test)
# %%
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')

# %%
# Save the model
import pickle
filepath = "/Users/kuldeeppatel/Documents/Load_digit_RandomForest.pkl"

with open(filepath, "wb") as file:
    pickle.dump(model, file)
print(f"Model saved to {filepath}")
# %%
#load the model
with open(filepath, 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")

# %%
predict_load_model = loaded_model.predict(X[:5])
# %%
loaded_model.score(X_test, y_test)
# %%
df.shape
# %%
