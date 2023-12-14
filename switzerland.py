import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Column names
columns = [
  'age', 'sex', 'cp', 'trestbps', 'chol',
  'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
  'slope', 'ca', 'thal', 'num'
]

# Read dataset
df = pd.read_csv("processed.switzerland.data", names=columns, header=None)

# Replacing '?' with 'NaN'
df.replace('?', np.nan, inplace=True)

# Drop column that have >50 row null value
for column in df.columns:
  if df[column].isnull().sum() > 50:
    df.drop(column, axis=1, inplace=True)

# Drop insignificant column
df.drop('chol', axis=1, inplace=True)

updated_df = df.dropna()

# Change the data type for each column
for column in updated_df.columns:
  if updated_df[column].dtypes == 'O':
    if column == "oldpeak":
      updated_df[column] = updated_df[column].astype(float)
    else:
      updated_df[column] = updated_df[column].astype(int)
  elif updated_df[column].dtypes == 'int64':
    updated_df[column] = updated_df[column].astype(int)

features = updated_df.drop('num', axis=1)
target = updated_df['num']

# Oversampling target 0, 2, 3, 4 based on target 1
oversample = SMOTE(k_neighbors=4)
X, y = oversample.fit_resample(features, target)

filled_df = X
filled_df['num'] = y

features = filled_df.drop('num', axis=1)
target = filled_df['num']

# Train - Test split
X_train ,X_test, y_train ,y_test = train_test_split(features, target, test_size = 0.2)

model = DecisionTreeClassifier()

accuracy_list = np.array([])

for i in range(0, 10):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracy = round((accuracy * 100), 2)

  accuracy_list = np.append(accuracy_list, accuracy)

min_accuracy = np.min(accuracy_list)
max_accuracy = np.max(accuracy_list)

# STREAMLIT
st.set_page_config(
  page_title = "Switzerland Heart Disease",
  page_icon = ":heart:"
)

st.title("Switzerland Heart Disease")
st.write("_Using Decision Tree Classifier_")
st.write(f"**Model's Accuracy**: :red[**{min_accuracy}%**] - :orange[**{max_accuracy}%**] (_Needs improvement! :red[Do not copy outright]_)")
st.write("\n")

age = st.number_input(label=":violet[**Age**]", min_value=filled_df['age'].min(), max_value=filled_df['age'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['age'].min()}**], :red[Max] value: :red[**{filled_df['age'].max()}**]")
st.write("")

sex_sb = st.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
st.write("")
st.write("")
if sex_sb == "Male":
  sex = 1
elif sex_sb == "Female":
  sex = 0
# -- Value 0: Female
# -- Value 1: Male

cp_sb = st.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
st.write("")
st.write("")
if cp_sb == "Typical angina":
  cp = 1
elif cp_sb == "Atypical angina":
  cp = 2
elif cp_sb == "Non-anginal pain":
  cp = 3
elif cp_sb == "Asymptomatic":
  cp = 4
# -- Value 1: typical angina
# -- Value 2: atypical angina
# -- Value 3: non-anginal pain
# -- Value 4: asymptomatic

trestbps = st.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=filled_df['trestbps'].min(), max_value=filled_df['trestbps'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['trestbps'].min()}**], :red[Max] value: :red[**{filled_df['trestbps'].max()}**]")
st.write("")

restecg_sb = st.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
st.write("")
st.write("")
if restecg_sb == "Normal":
  restecg = 0
elif restecg_sb == "Having ST-T wave abnormality":
  restecg = 1
elif restecg_sb == "Showing left ventricular hypertrophy":
  restecg = 2
# -- Value 0: normal
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach = st.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=filled_df['thalach'].min(), max_value=filled_df['thalach'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['thalach'].min()}**], :red[Max] value: :red[**{filled_df['thalach'].max()}**]")
st.write("")

exang_sb = st.selectbox(label=":violet[**Exercise induced angina**]", options=["No", "Yes"])
st.write("")
st.write("")
if exang_sb == "No":
  exang = 0
elif exang_sb == "Yes":
  exang = 1
# -- Value 0: No
# -- Value 1: Yes

oldpeak = st.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=filled_df['oldpeak'].min(), max_value=filled_df['oldpeak'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['oldpeak'].min()}**], :red[Max] value: :red[**{filled_df['oldpeak'].max()}**]")
st.write("")

slope_sb = st.selectbox(label=":violet[**Exercise induced angina**]", options=["Upsloping", "Flat", "Downsloping"])
st.write("")
st.write("")
if slope_sb == "Upsloping":
  slope = 1
elif slope_sb == "Flat":
  slope = 2
elif slope_sb == "Downsloping":
  slope = 3
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping

result = ":violet[-]"

st.write("")
if st.button("**Predict**", type="primary"):
  inputs = [[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope]]
  prediction = model.predict(inputs)[0]
  if prediction == 0:
    result = ":green[**Healthy**]"
  elif prediction == 1:
    result = ":orange[**Heart disease level 1**]"
  elif prediction == 2:
    result = ":orange[**Heart disease level 2**]"
  elif prediction == 3:
    result = ":red[**Heart disease level 3**]"
  elif prediction == 4:
    result = ":red[**Heart disease level 4**]"

st.write("")
st.write("")
st.subheader("Prediction:")
st.subheader(result)
