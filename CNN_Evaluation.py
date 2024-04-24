import chess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from Chess_Tools import tensorize

# Configuration
fileName = "eval_small_processed_chess_dataset.csv"
modelName = "ChessPrediction2.keras"

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

model = load_model(modelName)
model.summary()

# Evaluation: Model accuracy v steps away from final board state
stepCount = 20
accuracies = []
losses = []
label_encoder = LabelEncoder()

print("Step evaluations")
def doStepEvaluations():
    dfnames = ['FinalFEN']
    for step in range(stepCount):
        dfnames.append('LastFEN' + str(step + 1))
    print(dfnames)
    y = to_categorical(label_encoder.fit_transform(df['Result'].values))
    for dfname in dfnames:
        X = tensorize(df[dfname], df[dfname])
        loss, accuracy = model.evaluate(X, y, verbose=0)
        losses.append(loss)
        accuracies.append(accuracy)
    print("Losses:", losses)
    print("Accuracies:", accuracies)
doStepEvaluations()

plt.figure(figsize=(10, 6))
plt.plot(range(stepCount + 1), accuracies, marker='o', linestyle='-')
plt.xticks(range(stepCount + 1))
plt.title('Model Accuracy v. Steps Away from Final Board State')
plt.xlabel('Steps Away from Final Board State')
plt.ylabel('Model Accuracy')
plt.grid(True)
plt.show()

# Loss plotting
plt.figure(figsize=(10, 6))
plt.plot(range(stepCount + 1), losses, marker='o', linestyle='-')
plt.xticks(range(stepCount + 1))
plt.title('Model Loss v. Steps Away from Final Board State')
plt.xlabel('Steps Away from Final Board State')
plt.ylabel('Model Loss')
plt.grid(True)
plt.show()

# Predict on completely random data, as if we were making real predictions (FinalFEN <- RandomFEN)
print("Performing realistic test")
df2 = pd.read_csv('sm_realistic_processed_chess_dataset.csv')
X = tensorize(df2['RandomFEN'], df2['FinalFEN'])
y = to_categorical(label_encoder.fit_transform(df2['Result'].values))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=7)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Realistic test loss: {test_loss}, Realistic test accuracy: {test_accuracy}")