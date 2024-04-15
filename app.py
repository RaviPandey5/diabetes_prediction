import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return prediction



def main():
    st.title("Diabetes Predection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Predection App  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness = st.text_input("SkinThickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
    Age = st.text_input("Age")
    result=""
    try:
        if st.button("Predict"):
            result=predict_note_authentication(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
            if result == 1:
                st.text("Person Have Diabetes")
            else:
                st.text("Person Dont Have Diabetes")
    except:
        st.text("Error Please Provide Valid Inputs")
    #st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()