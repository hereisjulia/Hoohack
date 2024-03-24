## happy and health app 
## Streamlit Involve

import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("🌟 Welcome to Happy & Healthy!🌟")

st.write("")
st.write("Hey there, lovely soul! Welcome to our cozy corner of the digital world!😊")
st.write("Here, we're all about exploring emotions, tracking moods, and sprinkling a little extra sparkle into your day.")
st.write("Life can be a whirlwind sometimes, can't it? But fear not! Our app is here to be your trusty companion through it all. Whether you're feeling as bright as a sunny day 🌞 or as cozy as a rainy afternoon 🌧️, we've got your back!")
st.write("So go ahead, dive right in! Let's take a moment to tune into how you're feeling today. Log in, and together, let's embark on a journey of self-discovery and emotional well-being. 🚀✨")
st.write("We're thrilled to have you here! Let's make each day a little brighter, one mood at a time. Welcome aboard!💖")
st.divider()

st.subheader("Log In to track your mood")
with st.container():
    name = st.text_input("User Name")
    mail = st.text_input("Email")
    password = st.text_input("passwords")
    button1 = st.button("Sign in")

avatar_path = "E:/MQM_Courses/AllCase/Hoohack/animal-avatar-bear-svgrepo-com.svg"
with st.container(height=300) as messages:
    st.chat_message("H&H", avatar= avatar_path).write(f"Hello! {name}! Happy to see you here! ")
    emotion_score = st.chat_message("H&H", avatar= avatar_path).slider("How is your mood today?" , -2, 2)
    exercise_ = st.chat_message("H&H", avatar= avatar_path).radio("Have you exercise today?", ["Yes", "No"])
    stay_home_ = st.chat_message("H&H", avatar= avatar_path).radio("Have you left home today?", ["Yes", "No"])
    work_ = st.chat_message("H&H", avatar= avatar_path).radio("Did you work today?", ["Yes", "No"])
    talk_ = st.chat_message("H&H", avatar= avatar_path).slider("How many people you've talked to today? (choose 5 if more than 5)", 0, 5, 1)
    sleep_ = st.chat_message("H&H", avatar= avatar_path).slider("How long you've slept last night? (0 if none, 1 if less than 3 hours, 2 if less than 6 hours, 3 if more than 8 hours)", 0, 3, 1)
    relax_ = st.chat_message("H&H", avatar= avatar_path).radio("Did you do any relaxing activity today? (i.e. hang out with friend, try hobbies...", ["Yes", "No"])
    check = st.button("View Summary of my mood")


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
if check:
    st.subheader("Your mood today")
    st.table(data_today)
    st.divider()
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
    days_record = df.shape[0]
    st.header('Emotion Score Chart')
    st.write(f"Take a moment to explore your mood chart of the past {days_record} days with a sprinkle of positivity! ✨📊 Your emotional journey awaits! 💖")
    st.write("")
    st.line_chart(data = chart_data, x = 'Date', y = 'Scores', color='#dba495')
    st.divider()

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
    st.subheader("What are influencing your mood the most?")
    st.write("Discover the key factors shaping your mood over time! 🌟" )
    st.write("Reflect on your journey and gain insights into what brings you joy and fulfillment. Let's celebrate the highs and navigate through the lows together, embracing each experience as a stepping stone to greater happiness! ☀️💖")
    st.write("")
    st.bar_chart(rf_importance, x = 'Feature', y = 'Importance', color = "#dba495")
    selected_features = rf_importance.loc[rf_importance['Importance'] > 0.2, 'Feature']
