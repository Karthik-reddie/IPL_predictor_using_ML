import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit UI
st.title('IPL Win Predictor')

# Select boxes for teams and cities
batting_team = st.selectbox('Select the batting team', sorted(teams))
bowling_team = st.selectbox('Select the bowling team', sorted(teams))
selected_city = st.selectbox('Select host city', sorted(cities))

# Input fields for target, score, overs, and wickets
target = st.number_input('Target')
score = st.number_input('Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

# Predict Probability button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    win_prob, loss_prob = result[0][1], result[0][0]

    st.header(f'{batting_team} - {round(win_prob * 100)}%')
    st.header(f'{bowling_team} - {round(loss_prob * 100)}%')
