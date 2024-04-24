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
    vector = np.zeros((8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            value = piece_to_value[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            vector[row, col] = value
    return vector

def getMoveTensor(FEN):
    board = chess.Board()
    board.set_fen(FEN)
    movesFrom = np.zeros((8, 8), dtype=np.int8)
    movesTo = np.zeros((8, 8), dtype=np.int8)
    potentialMoves = board.legal_moves
    for move in potentialMoves:
        frrow, frcol = divmod(move.from_square, 8)
        torow, tocol = divmod(move.to_square, 8)
        movesFrom[frrow][frcol] = 1
        movesTo[torow][tocol] = 1
    return movesFrom, movesTo 

def getPieceTensor(FEN):
    board = chess.Board()
    board.set_fen(FEN)
    pieceMatrices = {}
    pieceTypes = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for piece in pieceTypes:
        pieceMatrices[piece] = np.zeros((8, 8), dtype=np.int8) 
    for square in chess.SQUARES:
        pieceOccupying = board.piece_at(square)
        if pieceOccupying:
            row, col = divmod(square, 8)
            ptype = pieceOccupying.piece_type
            pcolor = pieceOccupying.color
            val = 1
            if pcolor == chess.BLACK:
                val = -1
            pieceMatrices[ptype][row][col] = val
    return pieceMatrices[chess.PAWN], pieceMatrices[chess.KNIGHT], pieceMatrices[chess.BISHOP], pieceMatrices[chess.ROOK], pieceMatrices[chess.QUEEN], pieceMatrices[chess.KING]

def tensorize(fendf, findf):
    vector = np.array(fendf.apply(vectorize_board).tolist())
    move_tensors = fendf.apply(getMoveTensor).tolist()
    piece_tensors = fendf.apply(getPieceTensor).tolist()
    pawns, knights, bishops, rooks, queens, kings = zip(*piece_tensors)
    pawns = np.array(pawns)
    knights = np.array(knights)
    bishops = np.array(bishops)
    rooks = np.array(rooks)
    queens = np.array(queens)
    kings = np.array(kings)
    movesFrom, movesTo = zip(*move_tensors)
    movesFrom = np.array(movesFrom)
    movesTo = np.array(movesTo) 
    finvector = np.array(findf.apply(vectorize_board).tolist())
    final_move_tensors = findf.apply(getMoveTensor).tolist() 
    finmovesFrom, movesTo = zip(*move_tensors) 
    finmovesFrom = np.array(movesFrom) 
    finmovesTo = np.array(movesTo)
    finpiece_tensors = findf.apply(getPieceTensor).tolist()
    finpawns, finknights, finbishops, finrooks, finqueens, finkings = zip(*finpiece_tensors)
    finpawns = np.array(finpawns)
    finknights = np.array(finknights)
    finbishops = np.array(finbishops)
    finrooks = np.array(finrooks)
    finqueens = np.array(finqueens)
    finkings = np.array(finkings)
    return(np.stack((vector, movesFrom, movesTo, pawns, knights, bishops, rooks, queens, kings, finvector, finmovesFrom, finmovesTo, finpawns, finknights, finbishops, finrooks, finqueens, finkings), axis=1))