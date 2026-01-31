import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("logistic_model.pkl", "rb"))

st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
Age = st.number_input("Age", 1, 80, 25)
SibSp = st.number_input("Siblings/Spouses", 0, 5, 0)
Parch = st.number_input("Parents/Children", 0, 5, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Embarked", [0, 1, 2])

if st.button("Predict"):
    data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
