import pandas as pd

print("Reading csv...")

# Load the dataset
df = pd.read_csv('chess_games.csv')

print("Data successfully read. Columns:", df.columns)

print("Engineering features...")
# Engineer the ELO difference feature
df['elo_difference'] = df['WhiteElo'] - df['BlackElo']

# The opening ply is already provided in the dataset as 'opening_ply'
# Assuming 'overall_ply' can be derived from the 'AN' column (Algebraic Notation of the moves)
# Counting the number of moves in the 'AN' column to estimate overall ply
# This is a simplification and may not accurately count moves in all cases (e.g., castling, captures)
df['overall_ply'] = df['AN'].str.split().str.len()

print("Successfully engineered features.")

print("Dropping columns...")
# Remove columns that will not be used
columns_to_drop = ['Event', 'White', 'Black', 'UTCDate', 'UTCTime', 'WhiteRatingDiff', 'BlackRatingDiff']
df.drop(columns=columns_to_drop, inplace=True)
print("Successfully dropped columns.")

# Save the processed dataset to a new CSV file, if needed
print("Saving to new CSV file...")
df.to_csv('processed_chess_dataset.csv', index=False)
print("[DONE]")

print(df.head())
