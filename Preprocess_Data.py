import pandas as pd
import chess.pgn
import chess.svg
import io
import gc
import time
import random

# Config
fileName = 'chess_games.csv'
outputFile = True
outputFileName = 'eval_small_processed_chess_dataset.csv'
reductionFraction = .02
balanceDataset = False
LastMoveSave = 25

boardspacer = "\n- - - - - - - -" # Used for printing board state
startingBoardFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Load the dataset
pretime = time.time() # timestamp
print("Reading csv...")
df = pd.read_csv(fileName)

print("Data successfully read. Columns:", df.columns)

# Cut dataset in half
print("Reducing dataset size...")
df = df.sample(frac=reductionFraction)

print("Engineering features...")

# Only keep classical game rows
print("Filtering rows...")
games = df[(df['Event'] == ' Classical ') & (df['Result'] != '*')].copy()
games.reset_index(drop=True, inplace=True)
# Remove

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

# Balancing dataset
if balanceDataset:
    min_count = games['WhiteWin'].value_counts().min()
    games = pd.concat([
        games[games['WhiteWin'] == 1].sample(min_count, random_state=42),
        games[games['WhiteWin'] == 0.5],
        games[games['WhiteWin'] == 0].sample(min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

# Time Limit
print("Creating time limits...")
def getNumTimeLimit(str):
    return int(str.split('+')[0])
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

print("Saving FEN final board states & Ply Count...")
plyCounts = []
def gameToFen(gameAN):
    pgn = io.StringIO(gameAN)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    plyCounts.append(board.ply())
    return board.fen()

games["FinalFEN"] = games['AN'].apply(gameToFen)
games["PlyCount"] = plyCounts

lastMoves = {} # 0 is not included (0 is captured by FinalFEN), lastmove[x] is state that is x steps away from the final state
for m in range(LastMoveSave): 
    moveind = (m + 1)
    lastMoves[moveind] = []

print("Saving FEN random board states...")
def randomFens():
    randomFenList = []
    for i, row in games.iterrows():
        finmoves = []
        gameAN = row['AN']
        randMove = random.randint(1, row['PlyCount'])
        pgn = io.StringIO(gameAN)
        game = chess.pgn.read_game(pgn)
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            ply = board.ply()
            if ply == randMove:
                randomFenList.append(board.fen())
                if LastMoveSave == 0:
                    break
            lmstep = row['PlyCount'] - ply
            if lmstep > 0 and lmstep <= LastMoveSave:
                finmoves.append(board.fen())
        while len(finmoves) < LastMoveSave:
            finmoves.insert(0, startingBoardFEN)
        c = LastMoveSave
        for xstepinv in finmoves:
            lastMoves[c].append(xstepinv)
            c -= 1
    return randomFenList
games["RandomFEN"] = randomFens()

# Last moves
print("Saving last moves...")
for xstep in lastMoves:
    games["LastFEN" + str(xstep)] = lastMoves[xstep]

print("Successfully engineered features.")

# Save the processed dataset to a new CSV file, if needed
if outputFile:
    print("Saving to new CSV file...")
    games.to_csv(outputFileName, index=False)

posttime = time.time() # timestamp
print("[DONE] (" + str(round(posttime - pretime, 3)) + "s)")

print(games.head())
