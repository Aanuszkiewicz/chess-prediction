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

# My modules (shared between CNN_Improved and CNN_Evaluation)
from Chess_Tools import vectorize_board
from Chess_Tools import getMoveTensor
from Chess_Tools import getPieceTensor
from Chess_Tools import tensorize

# Configuration
fileName = "large_processed_chess_dataset.csv"
modelName = "ChessPrediction2.keras"
loadModel = False
epochs = 30

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

# Tensorizing data
print("Tensorizing data")
tensors = tensorize(df['RandomFEN'], df['FinalFEN'])
X = tensors
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(df['Result'].values))

# Data splitting
print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Model construction
model = None
hist = None
if not loadModel:
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(18, 8, 8), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=64, verbose=1, 
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    model.save(modelName)
else:
    model = load_model(modelName)
    model.summary()

# Model evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

origClasses = label_encoder.classes_ 
# Then you can map the indices to the actual class labels
for i, v in enumerate(origClasses):
    print(f"Index: {i}, Label: {v}")

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()