import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from scipy.stats import pearsonr
import statsmodels.api as sm
import time

# Configuration
fileName = "small_processed_chess_dataset.csv"

pretime = time.time() # timestamp

print(f"Reading {fileName}...")
df = pd.read_csv(fileName)

print("Creating opening dummies...")
ECO_dummies = pd.get_dummies(df['ECO'], drop_first=True, dtype=int)  # Convert 'ECO' into dummy variables
#X = pd.concat([df[['EloDiff']], ECO_dummies], axis=1)  # Concatenate 'EloDiff' and the dummy variables
X = pd.concat([df[['EloDiff']], df[['TimeLimit']], ECO_dummies], axis=1)  # Concatenate 'EloDiff' and the dummy variables
print(X)
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['WhiteWin']

# Fit the model
print("Fitting model...")
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Plotting the linear regression
plt.figure(figsize=(10, 6))

correlation, p_value = pearsonr(df['EloDiff'], df['WhiteWin'])
print(f"Correlation coefficient: {correlation}, P-value: {p_value}")

sns.regplot(x='EloDiff', y='WhiteWin', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('White Wins v. ELO Difference')
plt.xlabel('Elo Difference (White - Black)')
plt.ylabel('White Win Indicator (1 = Win, 0 = Loss/Tie)')

posttime = time.time() # timestamp
print("[DONE] (" + str(round(posttime - pretime, 3)) + "s)")

plt.show()