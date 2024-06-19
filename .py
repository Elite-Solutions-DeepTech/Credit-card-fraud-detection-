import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from google.colab import files
uploaded = files.upload()

data=pd.read_csv('Credit_Card_Fraud_Detection.csv')

data

print(data.shape)
df

print(df.isnull().sum)

data.dtypes
data.isna().sum()
data.info()

data.head(200)

x=data.iloc[:,:-1].values
y=data.iloc[:,13].values

x
y

from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
alg1=LogisticRegression()

alg1.fit(x_train,y_train)

y_pred=alg1.predict(x_test)

accuracy=alg1.score(x_train,y_train)

print(accuracy)

accuracy2=alg1.score(x_test,y_test)

print(accuracy2)

plt.scatter(y_test,y_pred,color='green')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red",linewidth=4)
plt.show()

#plot outputs
plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred,color='green',alpha=0.6)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red",linewidth=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted ')
plt.grid(True)
plt.show()