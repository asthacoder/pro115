#Uploading the csv
from google.colab import files
data_to_load = files.upload()

import pandas as pd
import plotly.express as px

df = pd.read_csv("115.csv")

velocity_list = df["Velocity"].tolist()
escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()

import pandas as pd
import plotly.express as px

df = pd.read_csv("escape_velocity.csv")

velocity_list = df["Velocity"].tolist()
escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(velocity_list, (len(velocity_list), 1))
Y = np.reshape(escaped_list, (len(escaped_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

# do hit and trial by changing the value of X_test
plt.axvline(x=X_test[23], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0, 30)
plt.show()

velocity = float(input("Enter the velocity:- "))
escape = model(velocity * lr.coef_ + lr.intercept_).ravel()[0]
if escape <= 0.01:
  print("The object will not escape the orbit")
elif escape >= 1:
  print("The object will escape the orbit")
elif escape < 0.5:
  print("The object might not escape the orbit")
else:
  print("The object might escape the orbit")