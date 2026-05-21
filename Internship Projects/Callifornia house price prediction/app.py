import gradio as gr
import pandas as pd
import numpy as np
import pickle 

with open('house_cost_xg_pipeline.pkl','rb') as f:
    model = pickle.load(f)

def pred_house_cost(longitude, latitude, housing_median_age, total_rooms,
                    total_bedrooms, population, households, median_income,ocean_proximity):

    input_df = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income, ocean_proximity
    ]], columns=[
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income','ocean_proximity'
    ])

    prediction = model.predict(input_df)[0]
    return f"House Price: {np.clip(prediction)}"

inputs = [
    gr.Number(label="Longitude"),
    gr.Number(label="Latitude"),
    gr.Number(label='Housing age'),
    gr.Number(label='Total rooms'),
    gr.Number(label='Total bedrooms'),
    gr.Number(label='Population in locality'),
    gr.Number(label='Households'),
    gr.Number(label='Median Income'),
    gr.Dropdown(['<1H OCEAN','INLAND','ISLAND','NEAR OCEAN'], label='Ocean proximity')
]

app = gr.Interface(
    fn=pred_house_cost,
    inputs=inputs,
    outputs="text",
    title="House Cost Predictor (California)"
)

app.launch(share=True)