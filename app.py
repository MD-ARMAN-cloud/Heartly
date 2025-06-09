import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import base64
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots
import matplotlib.pyplot as plt


st.title("Heart Disease Predictor")
tab1,tab2 = st.tabs(['Enter the detials (All are compulsory for checkup)',"Prediction in Batches"])

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
          st.markdown("""
      <h4 style='color: black;background-color: cyan;border-radius:10%;'>No Heart Disease Detected</h4>
  """, unsafe_allow_html=True)
       else:
          st.markdown("""
      <h4 style='color: black;background-color: cyan;border-radius:10%;'>Heart Disease Detected</h4>
  """, unsafe_allow_html=True)
       st.markdown('-------------------------')

# Function to create a download Link for a Dataframe as a CSV file
def get_binary_file_downloader_html(df):
  csv = df. to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  href = f'<a href="data:file/csv; base64, {b64}" download="predictions.csv">Download Predictions CSV</a>'
  return href


with tab2:
   st.title("Upload CSV File")
   st.subheader(' Instructions to note before uploading the file: ')
   st.info("""
1. No NaN values allowed.
2. Total 11 features in this order ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca','thal','BMI' ) . \n
3. Check the spellings of the feature names.
4. Feature values conventions: \n
- age: age of the patient [years] \n
- sex: sex of the patient [0: Male, 1: Female] \n
- cp: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2:Asymptomatic
- trestbps: resting blood pressure [mm Hg] \n
- chol: serum cholesterol [mm/dl] \n
- fbs: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
- restecg: resting electrocardiogram results [0: Normal, 1: abnormal)
- thalach: maximum heart rate achieved [Numeric value between 60 and 202] \n
- exang: exercise-induced angina [1: Yes, 0: No] \n
- oldpeak: oldpeak = ST [Numeric value measured in depression] \n
- slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n
- ca : number of major coloured fluoroscopy ( 0,1,2,3,4)\n
- thal : Thalasssemia("normal","Fixed defect","Reversible defect")
- BMI : Body mass index
""")

   # Create a file uploader in the sidebar
   uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
   if uploaded_file is not None:

      # Read the uploaded CSV file into a DataFrame
      input_data = pd.read_csv(uploaded_file)
      model = pickle.load(open('./RFC.pkl','rb'))
      expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca','thal','BMI']
      if set(expected_columns) .issubset(input_data.columns):
        new_column_names = {
        'age': 'Age',
        'sex': 'Sex',
        'cp': 'ChestPainType',
        'trestbps': 'RestingBloodPressure',
        'chol': 'SerumCholesterol',
        'fbs': 'FastingBloodSugar',
        'restecg': 'RestingECG',
        'thalach': 'MaxHeartRate',
        'exang': 'ExerciseInducedAngina',
        'oldpeak': 'STDepression',
        'slope': 'SlopeSTSegment',
        'ca': 'NumMajorVessels',
        'thal': 'Thalassemia'
           }

         # Rename the columns
        input_data.rename(columns=new_column_names, inplace=True)

        # Print first 5 rows of our data
        input_data.head()
        categorical_vars = ['ChestPainType', 'SlopeSTSegment','Thalassemia','NumMajorVessels']
        input_data[categorical_vars] = input_data[categorical_vars]. astype(str)
        input_data.replace('nan', np.nan, inplace=True)
        dummies = pd.get_dummies(input_data[categorical_vars], prefix=categorical_vars,dummy_na = False, drop_first=True).astype(np.int64)
        input_data = pd.concat([input_data, dummies], axis=1)
        input_data = input_data.drop(columns=categorical_vars)
        columns_to_scale = ['Age', 'RestingBloodPressure', 'SerumCholesterol', 'MaxHeartRate','STDepression','BMI'] # we have taken these columns for scale down
        input_data[columns_to_scale] = standardScaler.fit_transform(input_data[columns_to_scale])

        input_data['Prediction RFC']=''

        for i in range(len(input_data)):
          arr = input_data.iloc[i, :- 1].values
          input_data['Prediction RFC'][i] = model. predict([arr])[0]
        input_data.to_csv('PredictedHeartLR.csv')
        yes_count = (input_data['Prediction RFC'] == 1).sum()  # Count heart disease cases
        no_count = (input_data['Prediction RFC'] == 0).sum()  # Count no disease cases

        # Display the predictions
        st. subheader("Predictions:")
        st.write(input_data)
        st.markdown(get_binary_file_downloader_html(input_data),unsafe_allow_html=True)

        labels = ['Heart Disease', 'No Heart Disease']
        sizes = [yes_count, no_count]
        colors = ['red', 'green']

        # Create the Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True, startangle=140)
        ax1.set_title("Heart Disease Prediction Results")
        st.pyplot(fig1)

        # Create the Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, sizes, color=colors)
        ax2.set_xlabel("Condition")
        ax2.set_ylabel("Number of Patients")
        ax2.set_title("Heart Disease Prediction Results")
        st.pyplot(fig2)
        



      else:
        st.warning("Please make sure the uploaded CSV file has the correct columns.")
   else :
    st.info("Upload a CSV file to get predictions.")


# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed by MD ARMAN</p>", unsafe_allow_html=True)
