# %%
import numpy
import pandas as pd
import matplotlib as plt
import sklearn

data = pd.read_csv(r"C:\Users\Nischala\Downloads\Iris.csv")
print(data.head(10))
print("Count of missing values per column")
print(data.isnull().sum())  

# Histogram of sepal length
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(data['SepalLengthCm'], bins=20, color='skyblue', edgecolor='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length')
plt.show()

# Histogram of sepal length
plt.figure(figsize=(8, 6))
plt.hist(data['PetalLengthCm'], bins=20, color='skyblue', edgecolor='k')
plt.xlabel('Petal Length Cm')
plt.ylabel('Frequency')
plt.title('Distribution of Petal Length')
plt.show()

from sklearn.model_selection import train_test_split

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("total number of samples : ",len(data))
print("number of samples in training data : ",len(X_train))

##LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=500)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("accuracy score using logistic regression : ",accuracy_score(y_pred,y_test),'%')
ans = model.predict([[0,5.1, 3.5, 1.4, 0.2],
                     [1,6.7, 3.0, 5.2, 2.3]])
print(ans)

##DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
from sklearn.metrics import accuracy_score
print("accuracy score using logistic regression : ",accuracy_score(y_pred,y_test)*100,'%')


