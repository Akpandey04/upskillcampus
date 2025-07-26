import streamlit as st
import pandas as pd
import joblib
import holidays
from datetime import datetime

st.set_page_config(page_title="Traffic Forecaster", layout="wide", page_icon="🚗")

st.markdown("""
<style>
    div.stButton > button:first-child {
        border: 2px solid #4CAF50;
        color: white;
    }
    div.stButton > button:first-child:hover {
        border: 2px solid #f44336;
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

try:
    model = joblib.load('traffic_model_rf.pkl')
except FileNotFoundError:
    st.error("Model file not found! Make sure 'traffic_model_rf.pkl' is in the same folder as this app.")
    st.stop()

left_space, main_content, right_space = st.columns([1, 2, 1])

with main_content:
    st.title("Smart City Traffic Forecaster 🚗")
    st.write("This app predicts the number of vehicles at a junction for a given date and time.")
    st.markdown("---")

    result_placeholder = st.empty()

    st.header("Enter the details for prediction")

    date_input = st.date_input(
        "Select a Date",
        help="Choose the date for which you want to predict traffic."
    )

    time_input = st.time_input(
        "Select a Time",
        help="Choose the time for the prediction."
    )

    junction_input = st.selectbox(
        "Select a Junction",
        [1, 2, 3, 4],
        help="Choose the city junction number (1-4)."
    )

    st.write("")

    if st.button("Predict Traffic Volume", use_container_width=True):

        dt_input = datetime.combine(date_input, time_input)

        user_data = pd.DataFrame({
            'DateTime': [dt_input],
            'Junction': [junction_input]
        })


        def create_time_features(df):
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df['hour'] = df['DateTime'].dt.hour
            df['day'] = df['DateTime'].dt.day
            df['dayofweek'] = df['DateTime'].dt.dayofweek
            df['month'] = df['DateTime'].dt.month
            df['year'] = df['DateTime'].dt.year
            df['dayofyear'] = df['DateTime'].dt.dayofyear
            df['weekofyear'] = df['DateTime'].dt.isocalendar().week.astype(int)

            current_year = df['year'].iloc[0]
            india_holidays = holidays.country_holidays('IN', years=[current_year])
            df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: x in india_holidays).astype(int)
            df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
            return df


        processed_data = create_time_features(user_data)

        processed_data = pd.get_dummies(processed_data, columns=['Junction'])

        for j in [1, 2, 3, 4]:
            if f'Junction_{j}' not in processed_data.columns:
                processed_data[f'Junction_{j}'] = 0

        model_features = model.feature_names_in_
        processed_data = processed_data[model_features]

        prediction = model.predict(processed_data)

        with result_placeholder.container():
            st.success(f"**Prediction Result**")
            st.metric(label="Predicted Number of Vehicles", value=int(prediction[0]))
            st.markdown("---")
