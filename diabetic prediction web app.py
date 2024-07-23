import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
    #change the inputdata to numpy array
    input_data_to_array = np.asarray(input_data)

    #reshape the array for one instance
    data_reshape = input_data_to_array.reshape(1,-1)

    #prediction
    pred = loaded_model.predict(data_reshape)
    print(pred)

    if pred[0] == 0 :
      return "The person is not diabetic"
    else :
      return "The person is diabetic"

def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
