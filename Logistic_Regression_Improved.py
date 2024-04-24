import chess
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
fileName = "large_processed_chess_dataset.csv"

pretime = time.time() # timestamp

def vectorize_board(FEN):
    board = chess.Board()
    board.set_fen(FEN)
    piece_to_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 10
    }
    vector = np.zeros(64, dtype=int)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_to_value[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            vector[square] = value
    return vector

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

vectors = df["FinalFEN"].apply(vectorize_board)
vectorFrame = pd.DataFrame(vectors.tolist(), columns=['Square' + str(i + 1) for i in range(64)]).astype('int8')
print(vectors)
print(vectorFrame)

print("Creating opening dummies...")
ECO_dummies = pd.get_dummies(df['ECO'], drop_first=True, dtype=int).astype('int8')

print("Creating X...")
X = pd.concat([df[['EloDiff']], df[['TimeLimit']], ECO_dummies, vectorFrame], axis=1)
y = df["Result"]
print(X.head())
print()

print("Splitting data")
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=7)

print("Scaling data")
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

print("Fitting Model")
logreg = LogisticRegression(random_state=7, max_iter=10000)
logreg.fit(XTrainScaled, yTrain)

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

