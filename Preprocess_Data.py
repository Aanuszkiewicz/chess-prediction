import pandas as pd
import chess.pgn
import chess.svg
import io
import gc
import time

boardspacer = "\n- - - - - - - -" # Used for printing board state

# Load the dataset
pretime = time.time() # timestamp
print("Reading csv...")
df = pd.read_csv('chess_games.csv')

print("Data successfully read. Columns:", df.columns)

# Cut dataset in half
print("Reducing dataset size...")
df = df.sample(frac=1)

print("Engineering features...")

# Only keep classical game rows
print("Filtering rows...")
games = df[df['Event'] == ' Classical '].copy()
games.reset_index(drop=True, inplace=True)

# Delete original dataframe and clear up memory
print("Deleting original dataframe and garbage collecting...")
del df
gc.collect()

print("Dropping columns...")
# Remove unneccesary columns
columns_to_drop = ['Event', 'White', 'Black', 'UTCDate', 'UTCTime', 'WhiteRatingDiff', 'BlackRatingDiff']
games.drop(columns=columns_to_drop, inplace=True)

# ELO difference
print("Calculating ELO difference...")
games['EloDiff'] = games['WhiteElo'] - games['BlackElo']

# Win indicator (1 = white win, 0.5 = tie, 0 = black win)
print("Creating win indicator...")
games['WhiteWin'] = games['Result'].apply(lambda x: 1 if x == '1-0' else (0.5 if x == '1/2-1/2' else 0)) 

# Time Limit
print("Creating time limits...")
def getNumTimeLimit(str):
    return int(str.split("+")[0])
games['TimeLimit'] = games['TimeControl'].apply(getNumTimeLimit)

# # Count total number of moves made (more time needed)
# print("Calculating total ply...")
# games['TotalPly'] = 0
# for i in range(len(games)):
#     moves = games.at[i, 'AN']
#     pgn = io.StringIO(moves)
#     game = chess.pgn.read_game(pgn)
#     ply = game.end().board().ply()
#     games.at[i, 'TotalPly'] = ply

print("Successfully engineered features.")

# Save the processed dataset to a new CSV file, if needed
print("Saving to new CSV file...")
games.to_csv('processed_chess_dataset.csv', index=False)

posttime = time.time() # timestamp
print("[DONE] (" + str(round(posttime - pretime, 3)) + "s)")

print(games.head())
