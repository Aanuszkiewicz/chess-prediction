import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
import time

# Configuration
fileName = "processed_chess_dataset.csv"

pretime = time.time() # timestamp

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

print("Creating opening dummies...")
ECO_dummies = pd.get_dummies(df['ECO'], drop_first=True, dtype=int)  # Convert 'ECO' into dummy variables
X = pd.concat([df[['EloDiff']], df[['TimeLimit']], ECO_dummies], axis=1)  # Concatenate 'EloDiff' and the dummy variables
print(X)
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['WhiteWin']

# Fit the model
print("Fitting model...")
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Other interesting statistics
print("Computing game outcome percentages...")
mappedWins = df['WhiteWin'].map( {1: "White Wins", 0: "Black Wins", 0.5: "Tie"} )
winOutcomes = mappedWins.value_counts(normalize=True) * 100
print(winOutcomes)
# Generate bar chart
plt.figure(figsize=(8, 6))
winOutcomes.plot(kind = "bar", color = "mediumslateblue") 
plt.title("Chess Game Outcomes")
plt.ylabel("Percentage")
plt.xlabel("")
plt.xticks(rotation=0)
plt.yticks(range(0, 51, 5))
plt.grid(True, axis='y')
plt.show()

# Plotting the linear regressions
def linReg(xname, yname, params):
    dfx = df[xname]
    dfy = df[yname]
    r, p = pearsonr(dfx, dfy)
    r2 = pow(r, 2)
    print(f"r = {r}, r^2 = {r2}, p = {p}")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=xname, y=yname, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(params["title"])
    plt.xlabel(params["xlabel"])
    plt.ylabel(params["ylabel"])
    plt.show()

print("Plotting single variable regressions...")
linReg("EloDiff", "WhiteWin", {"title": "White Wins v. ELO Difference",
                                       "xlabel": "Elo Difference (White - Black)",
                                       "ylabel": "White Win Indicator (1 = Win, 0.5 = Tie, 0 = Loss"})
df["BlackWin"] = df['WhiteWin'].map( {1: 0, 0: 1, 0.5: 0.5} )
df["BlackEloDiff"] = df['EloDiff'].apply(lambda x: -x) 
linReg("BlackEloDiff", "BlackWin", {"title": "Black Wins v. ELO Difference",
                                       "xlabel": "Elo Difference (Black - White)",
                                       "ylabel": "Black Win Indicator (1 = Win, 0.5 = Tie, 0 = Loss"})

posttime = time.time() # timestamp
print("[DONE] (" + str(round(posttime - pretime, 3)) + "s)")