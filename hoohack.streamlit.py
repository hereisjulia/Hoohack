## happy and health app 
## Streamlit Involve

import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt

st.title("Welcome to H&H!")
st.write("Explore your inner self hehe.")

name = st.text_input("User Name")
email = st.text_input("Email")
password = st.text_input("passwords")
button1 = st.button("Sign in")


if button1:
    messages = st.container(height=300)
    messages.chat_message("H&H").write(f"Hello! {name}")
    messages.chat_message("H&H").write("Happy to see you here! I want to ask you some questions!")
    emotion_score = messages.chat_message("H&H").slider("How is your mood today?" , -2, 2, 0)
    exercise_ = messages.chat_message("H&H").radio("Have you exercise today?", ["Yes", "No"])
    stay_home_ = messages.chat_message("H&H").radio("Have you left home today?", ["Yes", "No"])
    work_ = messages.chat_message("H&H").radio("Did you work today?", ["Yes", "No"])
    talk_ = messages.chat_message("H&H").slider("How many people you've talked to today? (choose 5 if more than 5)", 0, 5, 1)
    sleep_ = messages.chat_message("H&H").slider("How long you've slept last night? (0 if none, 1 if less than 3 hours, 2 if less than 6 hours, 3 if more than 8 hours)", 0, 5, 1)
    relax_ = messages.chat_message("H&H").radio("Did you do any relaxing activity today? (i.e. hang out with friend, try hobbies...", ["Yes", "No"])
    check = st.button("Next")

data_today = {
    'Date': datetime.now().date(),
    'Scores': emotion_score,
    'Exercise': exercise_,
    'Stay_at_home': stay_home_,
    'Work/Study': work_,
    'Talk_to_people': talk_,
    'Sleep_time': sleep_,
    'Relax': relax_
}
data_today = pd.DataFrame(data_today, index = [0])
data_today.replace({'Yes': 1, 'No': 0}, inplace=True)

n_days = 100
#start_date = (datetime.now() - timedelta(days=n_days+1)).date()
start_date = datetime.now().date() - timedelta(days=n_days+1)
date_list = []
for i in range(n_days):
    new_date = start_date + timedelta(days=i)
    date_list.append(new_date)


random.seed(123)
scores = [random.randint(-2, 2) for _ in range(n_days)]
exercise = [random.randint(0, 1) for _ in range(n_days)]
stay_at_home = [random.randint(0, 1) for _ in range(n_days)]
work = [random.randint(0, 1) for _ in range(n_days)]
talk_to_people = [random.randint(0, 5) for _ in range(n_days)]
sleep_time = [random.randint(0, 3) for _ in range(n_days)]
relax = [random.randint(0, 1) for _ in range(n_days)]

data = {
    'Date': date_list,  # Example date range
    'Scores': scores,
    'Exercise': exercise,
    'Stay_at_home': stay_at_home,
    'Work/Study': work,
    'Talk_to_people': talk_to_people,
    'Sleep_time': sleep_time,
    'Relax': relax
}
df = pd.DataFrame(data)
df = pd.concat([df, data_today], axis=0)

# Print the DataFrame
chart_data = pd.DataFrame({
    'Date': df['Date'],
    'Scores': df['Scores']})

if check:
    st.subheader('Emotion Score Chart')
    st.line_chart(data = chart_data, x = 'Date', y = 'Scores')


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

if check:
    st.bar_chart(rf_importance, x = 'Feature', y = 'Importance')

