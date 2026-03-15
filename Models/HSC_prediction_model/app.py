import gradio as gr
import numpy as np
import pandas as pd
import pickle

# load the model
with open('student_rf_pipeline.pkl' ,'rb') as file:
    model = pickle.load(file)

# main logic
def predict_gpa(gender , age , address, famsize, Pstatus,M_Edu,F_Edu,M_Job,F_Job,
                relationship , smoker,tuition_fee, time_friends,ssc_result ):
    input_df = pd.DataFrame([[
        gender , age , address, famsize, Pstatus,M_Edu,F_Edu,M_Job,F_Job,
        relationship , smoker,tuition_fee, time_friends,ssc_result
    ]],
      
    columns=[
        'gender' , 'age' , 'address', 'famsize', 'Pstatus','M_Edu','F_Edu','M_Job','F_Job',
        'relationship' , 'smoker','tuition_fee', 'time_friends','ssc_result'
    ]                      
                            
    )
    
    # predict
    prediction = model.predict(input_df)[0]
    
    # clipping
    return f'Predicted HSC result: {np.clip(prediction,0,5):.2f}'

input = [
    gr.Radio(['M','F'], label='Gender'),
    gr.Number(label='Age' , value = 18),
    gr.Radio(['Urban' , 'Rural'], label='Address'),
    gr.Radio(['GT3','LE3'], label='Family Size'),
    gr.Radio(['Together','Apart'],label='parent status'),
    gr.Slider(0,4 , step=1, label="Mother's edu"),
    gr.Slider(0,4 , step=1, label="Father's edu"),
    gr.Dropdown(['Teacher','Health','Services','At_home','Other'],label="Mother's job"),
    gr.Dropdown(['Farmer','Teacher','Business','Health','Services','Other'],label="Father's job"),
    gr.Radio(['Yes','No'],label='Relationship'),
    gr.Radio(['Yes','No'],label='Smoker'),
    gr.Number(label='Tuition fee'),
    gr.Slider(1,5 ,step=1 , label="Time with fridends(hours)"),
    gr.Number(label='SSC result (gpa)')
    
]

# interface 
app = gr.Interface(
    fn = predict_gpa,
    inputs=input,
    outputs='text',
    title='Hsc result predictor'
)

# launch
# app.launch()
app.launch(share=True)