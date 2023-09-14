import streamlit as st
import joblib
import pandas as pd

# loading the trained model
model = joblib.load('XGBClassifier.pkl')

st.title('XGBoost Classifier')

feature1 = st.sidebar.slider('feature1', 10.59, 21.18, 13.0)
feature2 = st.sidebar.slider('feature2', 12.41, 17.25, 13.0)
feature3 = st.sidebar.slider('feature3', 0.8081,0.9183,  0.9)
feature4 = st.sidebar.slider('feature4', 4.899, 6.675, 4.9)
feature5 = st.sidebar.slider('feature5', 2.63, 4.033, 3.0)
feature6 = st.sidebar.slider('feature6',0.7651, 8.456, 3.0)
feature7 = st.sidebar.slider('feature7', 4.519, 6.55, 5.0)

df = pd.DataFrame({'0': feature1, '1': feature2, '2': feature3, '3': feature4, '4': feature5, '5': feature6, '6': feature7}, index=[0])

model_prediction = model.predict(df)

st.header('Prediction')
st.write(df, model)
prediction = pd.DataFrame({'Prediction': model_prediction})
st.write(prediction)


