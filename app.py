import streamlit as st
import pandas as pd
import time
from ml_model import SyntheticDataGenerator
import altair as alt


sensor_names = ["Temperature Sensor", "Pressure Sensor", "Vibration Sensor", "Humidity Sensor"]
generator = SyntheticDataGenerator(sensor_names)

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame(columns=["timestamp"] + sensor_names)

def update_sensor_data(new_data):
    st.session_state.sensor_data = pd.concat([st.session_state.sensor_data, pd.DataFrame([new_data])], ignore_index=True)

st.sidebar.title("Sensor Thresholds")
thresholds = {sensor: st.sidebar.slider(sensor, 0, 100, 70) for sensor in sensor_names}

st.title("Predictive Maintenance Dashboard")
st.subheader("Real-time Sensor Data")

chart_placeholders = {sensor: st.empty() for sensor in sensor_names}
alert_placeholders = {sensor: st.empty() for sensor in sensor_names}

def create_line_chart(data, sensor):
    chart = alt.Chart(data).mark_line().encode(
        x='timestamp:T',
        y=f'{sensor}:Q'
    ).properties(
        width=600,
        height=300,
        title=f'{sensor} Readings'
    )
    return chart

while True:
    sensor_data = generator.generate_data()
    update_sensor_data(sensor_data)

    st.session_state.sensor_data['timestamp'] = pd.to_datetime(st.session_state.sensor_data['timestamp'])

    for sensor in sensor_names:
        chart_placeholders[sensor].altair_chart(create_line_chart(st.session_state.sensor_data, sensor))

        if sensor_data[sensor] > thresholds[sensor]:
            alert_placeholders[sensor].error(f"{sensor} has crossed the threshold with a reading of {sensor_data[sensor]:.2f}")
        else:
            alert_placeholders[sensor].empty()  

    time.sleep(1)  
