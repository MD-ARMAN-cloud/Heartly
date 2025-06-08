import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import base64
from sklearn.preprocessing import StandardScaler

st.title("Heart Disease Predictor")
tab1,tab2 = st.tabs(['Enter the detials (All are compulsory for checkup) ',"."])

with tab1:
  Age = st.number_input("Age (years)", min_value=0, max_value=150)
  Sex = st. selectbox("Sex", ["Male", "Female"])
  ChestPainType = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain","Asymptomatic"])
  RestingBloodPressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
  SerumCholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
  FastingBloodSugar = st. selectbox("Fasting Blood Sugar", [" <= 120 mg/dl", "> 120 mg/dl"])
  RestingECG = st.selectbox("Resting ECG Results", ["Normal", "Abnormal"])
  MaxHeartRate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
  ExerciseInducedAngina = st. selectbox("Exercise-Induced Angina", ["Yes", "No"])
  STDepression = st.number_input("Oldpeak    (ST Depression)", min_value=0.0, max_value=10.0)
  SlopeSTSegment = st. selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping" ] ) 
  NumMajorVessels=st.selectbox("Number of major blood vessels",["0","1","2","3","4"])
  Thalassemia=st.selectbox("Thalassemia level",["normal","Fixed defect","Reversible defect","hyper reversible defect"])
  BMI=st.number_input("Body Mass Index",min_value=0)


Sex = 0 if Sex == "Male" else 1
ChestPainType = [ "Typical Angina","Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(ChestPainType)
FastingBloodSugar = 1 if FastingBloodSugar == "> 120 mg/dl" else 0
RestingECG = ["Normal", "Abnormal"].index(RestingECG)
ExerciseInducedAngina = 1 if ExerciseInducedAngina == "Yes" else 0
SlopeSTSegment = ["Upsloping", "Flat", "Downsloping"].index(SlopeSTSegment)
NumMajorVessels = ["0","1","2","3","4"].index(NumMajorVessels)
Thalassemia = ["normal","Fixed defect","Reversible defect"].index(Thalassemia)


# Create a DataFrame with user inputs
input_data = pd. DataFrame({
'Age': [Age],
'Sex' : [Sex],
'RestingBloodPressure' : [RestingBloodPressure],
'SerumCholesterol': [SerumCholesterol],
'FastingBloodSugar' : [FastingBloodSugar],
'RestingECG': [RestingECG],
'MaxHeartRate': [MaxHeartRate],
'ExerciseInducedAngina': [ExerciseInducedAngina],
'STDepression': [STDepression],
'BMI':[BMI]
})

ChestPainType_1=0
ChestPainType_2=0
ChestPainType_3=0

if ChestPainType == 1:  
    ChestPainType_1 = 1
elif ChestPainType == 2:  
    ChestPainType_2 = 1
elif ChestPainType == 3:  
    ChestPainType_3 = 1

input_data['ChestPainType_1'] =ChestPainType_1
input_data['ChestPainType_2'] =ChestPainType_2
input_data['ChestPainType_3'] =ChestPainType_3

SlopeSTSegment_1 = 0
SlopeSTSegment_2 = 0

# Assign values based on index
if SlopeSTSegment == 1:  # Flat
    SlopeSTSegment_1 = 1
elif SlopeSTSegment == 2:  # Downsloping
    SlopeSTSegment_2 = 1

# Now, include them in your DataFrame
input_data['SlopeSTSegment_1'] = SlopeSTSegment_1
input_data['SlopeSTSegment_2'] = SlopeSTSegment_2

Thalassemia_1 =0
Thalassemia_2 = 0 
Thalassemia_3=0

if Thalassemia == 1:  
    Thalassemia_1 = 1
elif Thalassemia == 2:  
    Thalassemia_2 = 1
elif Thalassemia==3:
    Thalassemia_3=1


input_data['Thalassemia_1'] =Thalassemia_1
input_data['Thalassemia_2'] =Thalassemia_2
input_data['Thalassemia_3'] =Thalassemia_3

NumMajorVessels_1 =0
NumMajorVessels_2 =0
NumMajorVessels_3 =0
NumMajorVessels_4 =0

if NumMajorVessels == 1:  
    NumMajorVessels_1 = 1
elif NumMajorVessels == 2:  
    NumMajorVessels_2 = 1
elif NumMajorVessels == 3:  
    NumMajorVessels_3 = 1
elif NumMajorVessels == 4:  
    NumMajorVessels_4 = 1

input_data['NumMajorVessels_1'] =NumMajorVessels_1
input_data['NumMajorVessels_2'] =NumMajorVessels_2
input_data['NumMajorVessels_3'] =NumMajorVessels_3
input_data['NumMajorVessels_4'] =NumMajorVessels_4

standardScaler = StandardScaler()
columns_to_scale = ['Age', 'RestingBloodPressure', 'SerumCholesterol', 'MaxHeartRate','STDepression','BMI'] # we have taken these columns for scale down
input_data[columns_to_scale] = standardScaler.fit_transform(input_data[columns_to_scale])

algonames = ['Logistic Regression', 'Decision Tree', 'Random Forest Classifier', 'Support Vector Machine']
modelnames = [ 'LogisticsReg.pkl', 'DTree.pkl','RFC.pkl', 'SVM.pkl']

from sklearn.utils.validation import check_is_fitted
predictions = []
def predict_heart_disease(data):
  for modelname in modelnames:
    model = pickle. load(open(f"./{modelname}"   , 'rb'))
    check_is_fitted(model)
    prediction = model.predict(data)
    predictions. append(prediction)
  return predictions


# Create a submit button to make predictions
if st.button("Submit"):
  st.subheader('Results ...')
  st. markdown ('-----------------------')

  result = predict_heart_disease(input_data)
   
  for i in range(len(predictions)):
     st.subheader(algonames[i])
     if result[i][0]==0:
        st.write("No heart disease detected.")
     else:
        st.write("Heart disease detected.")
     st.markdown('-------------------------')

