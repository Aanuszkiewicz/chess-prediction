import pandas as pd

fileName = "large_processed_chess_dataset.csv"

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

meanPly = df['PlyCount'].mean()
ERandPly = meanPly / 2

print("Expected ply of random board state:", ERandPly)