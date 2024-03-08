import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Configuration
fileName = "small_processed_chess_dataset.csv"

pretime = time.time() # timestamp

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

print("Creating opening dummies...")
ECO_dummies = pd.get_dummies(df['ECO'], drop_first=True, dtype=int)
X = pd.concat([df[['EloDiff']], df[['TimeLimit']], ECO_dummies], axis=1)
y = df["Result"]
print(X.head())
print()

# Split the data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=7)

# Scale data
scaler = StandardScaler() #Transforms features to have a mean of 0 and a standard deviation of 1
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

# Fit logistic regression model
logreg = LogisticRegression(random_state=7, max_iter=10000)
logreg.fit(XTrainScaled, yTrain)

# yTrainPred = logreg.predict(XTrain)
# trainReport = classification_report(yTrain, yTrainPred)
# print(trainReport)

yPred = logreg.predict(XTestScaled)

print("Classes:", logreg.classes_)
print()
cnfMatrix = metrics.confusion_matrix(yTest, yPred)
print("Confusion Matrix:\n", cnfMatrix)

class_names=["Black Win", "White Win", "Tie"]

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
sns.heatmap(pd.DataFrame(cnfMatrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.xticks(tick_marks + 0.5, class_names)
plt.yticks(tick_marks + 0.5, class_names)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')

plt.text(0.5,257.44,'Predicted Class')

print()
print(classification_report(yTest, yPred, target_names=class_names))

plt.show()

posttime = time.time() # timestamp
print("[DONE] (" + str(round(posttime - pretime, 3)) + "s)")