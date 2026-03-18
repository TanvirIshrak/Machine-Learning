import gradio as gr
import pandas as pd
import numpy as np
import pickle


with open('medical_cost_gb_pipeline.pkl','rb') as file:
    model = pickle.load(file)
    
def predict_cost(age,sex,bmi,children,smoker,region):
    input_df = pd.DataFrame([[
        age,sex,bmi,children,smoker,region
    ]],
    
    columns=[
        'age','sex','bmi','children','smoker','region'
    ]
    
    )
    
    prediction = model.predict(input_df)[0]
    return f'Predicted cost : {prediction}'

input = [
    gr.Number(label='Age', value=18),
    gr.Radio(['M','F'], label='Gender'),
    gr.Number(label='BMI', value=25.0),
    gr.Slider(0,7 , step=1, label='Children'),
    gr.Radio(['Yes','No'], label='Smoker'),
    gr.Radio(['northwest','southwest','southeast','northeast'])
]

app = gr.Interface(
    fn = predict_cost,
    inputs=input,
    outputs='text',
    title='Medical cost predictor'
)

app.launch(share=True)