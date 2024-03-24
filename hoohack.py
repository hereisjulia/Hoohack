import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb

"""# Data Generating

Generating random data of a person.
"""

n_days = 200
start_date = (datetime.now() - timedelta(days=n_days)).date()

random.seed(123)

scores = [random.randint(-2, 2) for _ in range(n_days)]
exercise = [random.randint(0, 1) for _ in range(n_days)]
stay_at_home = [random.randint(0, 1) for _ in range(n_days)]
work = [random.randint(0, 1) for _ in range(n_days)]
talk_to_people = [random.randint(0, 5) for _ in range(n_days)]
sleep_time = [random.randint(0, 3) for _ in range(n_days)]
relax = [random.randint(0, 1) for _ in range(n_days)]

data = {
    'Date': pd.date_range(start= start_date, periods=n_days),  # Example date range
    'Scores': scores,
    'Exercise': exercise,
    'Stay_at_home': stay_at_home,
    'Work/Study': work,
    'Talk_to_people': talk_to_people,
    'Sleep_time': sleep_time,
    'Relax': relax
}

df = pd.DataFrame(data)

# Print the DataFrame
print(df)

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

## split data
df['const'] = 1
X = df[['Date', 'Exercise', 'Stay_at_home', 'Work/Study', 'Talk_to_people', 'Sleep_time', 'Relax', 'const']]  # Independent variables
y = df['Scores']            # Dependent variable

train_size = int(0.8 * len(df))

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

train_Date = X_train['Date']
test_Date = X_test['Date']

X.drop(columns=['Date'], inplace=True)
X_train.drop(columns=['Date'], inplace=True)
X_test.drop(columns=['Date'], inplace=True)

"""# Modeling

## Linear Regression
"""

import statsmodels.api as sm

# Fit the linear regression model
model = sm.OLS(y_train, X_train).fit()

# Print the summary of the model
print(model.summary())

print('Train R2:', r2_score(y_train, model.predict(X_train)))
print('Train MSE:', mean_squared_error(y_train, model.predict(X_train)))

plt.plot(train_Date, y_train, label='Scores')
plt.plot(train_Date, model.predict(X_train), label='LM')
plt.xlabel('Date')
plt.ylabel('Emotion Score')
plt.legend()

"""## Random Forest"""

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print('Train R2:', r2_score(y_train, rf_model.predict(X_train)))
print('Train MSE:', mean_squared_error(y_train, rf_model.predict(X_train)))

plt.plot(train_Date, y_train, label='Scores')
plt.plot(train_Date, rf_model.predict(X_train), label='RF')
plt.title('Train/Test')
plt.xlabel('Date')
plt.ylabel('Emotion Score')
plt.legend()

plt.plot(test_Date, y_test, label='Test Scores')
plt.plot(test_Date, rf_model.predict(X_test), label='Test RF')
plt.xlabel('Date')
plt.ylabel('Emotion Score')
plt.legend()

importance = rf_model.feature_importances_
feature_names = ['Exercise', 'Stay_at_home', 'Work/Study', 'Talk_to_people', 'Sleep_time', 'Relax', 'const']
rf_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
rf_importance

"""## XGBoost"""

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 3
}
num_round = 100
XGB_model = xgb.train(params, dtrain, num_round)

print('Train R2:', r2_score(y_train, XGB_model.predict(dtrain)))
print('Train MSE:', mean_squared_error(y_train, XGB_model.predict(dtrain)))

plt.plot(train_Date, y_train, label='Scores')
plt.plot(train_Date, XGB_model.predict(dtrain), label='XGB')
plt.xlabel('Date')
plt.ylabel('Emotion Score')
plt.plot(test_Date, y_test, label='Test Scores')
plt.plot(test_Date, XGB_model.predict(dtest), label='Test XGB')
plt.legend()