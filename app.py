import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gradio as gr
import os

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
df = pd.read_csv("seattle-weather.csv")

# Keep only required columns
df = df[['temp_max', 'temp_min', 'wind', 'weather']]

# Encode weather (string → numbers)
df['weather'] = df['weather'].astype('category').cat.codes

# Split features & label
X = df[['temp_max', 'temp_min', 'wind']]
y = df['weather']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Weather mapping
weather_map = {
    0: "drizzle",
    1: "fog",
    2: "rain",
    3: "snow",
    4: "sun"
}

# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------
def predict_weather(temp_max, temp_min, wind):
    data = np.array([[temp_max, temp_min, wind]])
    pred = model.predict(data)[0]
    return f"Predicted Weather: {weather_map[pred]}"

# -------------------------------------------------------
# GRADIO UI
# -------------------------------------------------------
ui = gr.Interface(
    fn=predict_weather,
    inputs=[
        gr.Number(label="Max Temperature (°C)"),
        gr.Number(label="Min Temperature (°C)"),
        gr.Number(label="Wind Speed (m/s)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🌤 Weather Forecasting App",
    description="Enter temperature and wind speed to predict weather condition."
)

# -------------------------------------------------------
# RENDER DEPLOYMENT SETTINGS
# -------------------------------------------------------
port = int(os.environ.get("PORT", 7860))

ui.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    inbrowser=False  # VERY IMPORTANT FOR RENDER
)
