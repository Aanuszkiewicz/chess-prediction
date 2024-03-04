import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv('processed_chess_dataset.csv')

# Plotting the linear regression
plt.figure(figsize=(10, 6))

# Creating an indicator variable
df['WhiteWin'] = df['Result'].apply(lambda x: 1 if x == '1-0' else (0.5 if x == '1/2-1/2' else 0)) 
#df['WhiteWin'] = df['Result'].apply(lambda x: 1 if x == '1-0' else 0)
#df['WhiteWin'] = df['Result'].apply(lambda x: 0 if x == '0-1' else 1)
print(df.head())

# # Selecting features
# X = df[['EloDiff']]  # A DataFrame with the features
# # Selecting target
# y = df['WhiteWin']  # A Series with the target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# reg = linear_model.LinearRegression()
# reg.fit(X_train, y_train)

# # Train the model using the training sets
# reg.fit(X_train, y_train)

# # Make predictions using the testing set
# win_y_pred = reg.predict(X_test)

# # The coefficients
# print("Coefficients: \n", reg.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, win_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, win_y_pred))

# # Plot outputs
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, win_y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

# Assuming 'EloDiff' is your independent variable and 'WhiteWin' is your dependent variable
# Add a constant to the independent variable to include an intercept in the model





# X = sm.add_constant(df['EloDiff'])  # Adds a constant term to the predictor
# y = df['WhiteWin']

# # Fit the model
# model = sm.OLS(y, X).fit()

# # Print the summary
# print(model.summary())

ECO_dummies = pd.get_dummies(df['Opening'], drop_first=True, dtype=int)  # Convert 'ECO' into dummy variables
print(ECO_dummies)
#X = pd.concat([df[['EloDiff']], ECO_dummies], axis=1)  # Concatenate 'EloDiff' and the dummy variables
X = pd.concat([df[['EloDiff']], ECO_dummies], axis=1)  # Concatenate 'EloDiff' and the dummy variables
print(X)
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['WhiteWin']

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())


correlation, p_value = pearsonr(df['EloDiff'], df['WhiteWin'])
print(f"Correlation coefficient: {correlation}, P-value: {p_value}")

sns.regplot(x='EloDiff', y='WhiteWin', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('White Wins v. ELO Difference')
plt.xlabel('Elo Difference (White - Black)')
plt.ylabel('White Win Indicator (1 = Win, 0 = Loss/Tie)')

plt.show()
