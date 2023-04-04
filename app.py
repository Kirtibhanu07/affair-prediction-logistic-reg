import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


st.title('Predicting Participation in Extramarital Affairs')
st.write("**Logistic Regression Model**")
st.write("---")
st.markdown("The dataset is the affairs dataset that comes with Statsmodels. It was derived from a survey of women in 1974 by Redbook magazine, in which married women were asked about their participation in extramarital affairs. More information about the study is available in a 1978 paper from the Journal of Political Economy.")
st.write("---")
affair = pd.read_pickle("affair.pkl")
model = pickle.load(open('logistic_model.pkl', 'rb'))
data = pd.DataFrame.from_dict(affair)

# Define options for selectboxes
rate_marriage_options = ["Very Poor", "Poor", "Average", "Good", "Very Good"]
religious_options = ["Not Religious", "Slightly Religious", "Moderately Religious", "Strongly Religious"]
educ_options = ["Grade School", "High School", "Some College", "College Graduate", "Some Graduate School", "Advanced Degree"]
occupation_options = ["Student", "Farming/Semi-Skilled/Unskilled", "White Collar", "Teacher/Nurse/Writer/Technician/Skilled", "Managerial/Business", "Professional with Advanced Degree"]

# Create dataframe
# Add inputs
inc = 1
st.sidebar.header("Fill out..")

age = st.sidebar.slider("Your Age", min_value=17, max_value=42, step=1)
yrs_married = st.sidebar.slider("Years Together", min_value=0, max_value=23, step=1)
rate_marriage = st.selectbox("Rate Your Relationship", options=rate_marriage_options)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, step=1)
education = st.selectbox("Level of Education", options=educ_options)
occupation = st.selectbox("Your Occupation", options=occupation_options)
occupation_husb = st.selectbox("Your Partner's Occupation", options=occupation_options)
religious = st.selectbox("How Religious Are You?", options=religious_options)
#affairs = st.selectbox("Have You Had an Affair before?", options=["No", "Yes"])

def education_type(education):
        if education == "Grade School":
                edu = 9
        elif education == "High School":
                edu = 12
        elif education == "Some College":
                edu = 14
        elif education == "College Graduate":
                edu = 16
        elif education == "Some Graduate School":
                edu = 17
        else:
                edu = 20
        return edu

def marriage_rating(rate_marriage):
        if rate_marriage == 'Very Poor':
                rate = 1
        elif rate_marriage == 'Poor':
                rate = 2
        elif rate_marriage == 'Average':
                rate = 3
        elif rate_marriage == 'Good':
                rate = 4
        else:
                rate = 5
        return rate

def rel(religious):
        if religious == "Not Religious":
                reli = 1
        elif religious == "Slightly Religious":
                reli = 2
        elif religious == "Moderately Religious":
                reli = 3
        else:
                reli = 4
        return reli

occ_2 = 1 if occupation == 'Farming/Semi-Skilled/Unskilled' else 0
occ_3 = 1 if occupation == 'White Collar' else 0
occ_4 = 1 if occupation == 'Teacher/Nurse/Writer/Technician/Skilled' else 0
occ_5 = 1 if occupation == 'Managerial/Business' else 0
occ_6 = 1 if occupation == 'Professional with Advanced Degree' else 0

occ_husb_2 = 1 if occupation_husb == 'Farming/Semi-Skilled/Unskilled' else 0
occ_husb_3 = 1 if occupation_husb == 'White Collar' else 0
occ_husb_4 = 1 if occupation_husb == 'Teacher/Nurse/Writer/Technician/Skilled' else 0
occ_husb_5 = 1 if occupation_husb == 'Managerial/Business' else 0
occ_husb_6 = 1 if occupation_husb == 'Professional with Advanced Degree' else 0

if occupation == 'Student':
    occ_2, occ_3, occ_4, occ_5, occ_6 = 0, 0, 0, 0,0
if occupation_husb == 'Student':
    occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5, occ_husb_6 = 0, 0, 0, 0, 0

rate_marriage = int(marriage_rating(rate_marriage))
age = int(age)
yrs_married = int(yrs_married)
children = int(children)
religious = int(rel(religious))
educ = int(education_type(education))
occ_2 = int(occ_2)
occ_3= int(occ_3)
occ_4 = int(occ_4)
occ_5 = int(occ_5)
occ_6 = int(occ_6)
occ_husb_2 = int(occ_husb_2)
occ_husb_3 = int(occ_husb_3)
occ_husb_4 = int(occ_husb_4)
occ_husb_5 = int(occ_husb_5)
occ_husb_6 = int(occ_husb_6)
Intercept = 1

feature_list = [Intercept, occ_2, occ_3, occ_4, occ_5, occ_6, occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5, occ_husb_6, rate_marriage, age, yrs_married, children, religious, educ]
predictions = model.predict_proba([feature_list]).reshape(1, -1)
pred = model.predict([feature_list]).reshape(1,-1)
affair_pred = int(pred[0][0])
result = round(predictions[0][1]*100,2)

st.write("----")
st.subheader("Results...")
st.write('Probaility of participating in an Extra Martial Affair   :   ',result,' %')
st.write('Affair Classification [1: Yes/ 0: No]    :   ',affair_pred)
#st.write(model.predict_proba(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 18, 1 , 0, 1, 16]])))