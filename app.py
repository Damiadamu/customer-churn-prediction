import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn import set_config

set_config(transform_output="pandas")


class CustomKMeans(KMeans):
    def get_feature_names_out(self):
        feats = np.concatenate((self.feature_names_in_, ["cluster_label"]))
        return feats

    def _transform(self, X):
        labels = self.predict(X).reshape(-1, 1)
        X_t = np.hstack((X, labels))
        return X_t

    def transform(self, X):
        return self._transform(X)


# Load the pre-trained model
model = joblib.load("./models/churn_model.pkl")

# Define the features your model was trained on
# This list must be in the exact order the model expects
FEATURES = [
    "creditscore",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "numofproducts",
    "hascrcard",
    "isactivemember",
    "estimatedsalary",
]


def predict_churn(inputs):
    """
    Takes a dictionary of user inputs and makes a churn prediction.
    """
    # Create a DataFrame from the inputs
    input_df = pd.Series(inputs).to_frame().T

    # Ensure the columns are in the correct order
    input_df = input_df[FEATURES]

    # Make a prediction
    prediction = model.predict(input_df)
    return prediction[0]


# --- Streamlit App Layout ---
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter the customer's details to predict if they will churn.")
st.write("---")

# Create input widgets for each feature
input_data = {}
st.header("Customer Information")

input_data["creditscore"] = st.number_input(
    "Credit Score", min_value=350, max_value=850
)
input_data["country"] = st.selectbox("Country", ["France", "Germany", "Spain"]).lower()
input_data["gender"] = st.selectbox("Gender", ["Male", "Female"]).lower()
input_data["age"] = st.number_input("Age", min_value=18)
input_data["tenure"] = st.number_input("Tenure", min_value=0)
input_data["balance"] = st.number_input("Balance", min_value=0)
input_data["numofproducts"] = st.number_input("Number of Products", min_value=0)
input_data["hascrcard"] = (
    1 if st.selectbox("Has Credit Card", ["Yes", "No"]).lower() == "yes" else 0
)
input_data["isactivemember"] = (
    1 if st.selectbox("Is Active Member", ["Yes", "No"]).lower() == "yes" else 0
)
input_data["estimatedsalary"] = st.number_input("Estimated Salary", min_value=0)

# Prediction button
st.write("---")
if st.button("Predict Churn"):
    # Make the prediction
    prediction = predict_churn(input_data)

    # Display the result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.write("Prediction: This customer is likely to **churn**.")
    else:
        st.write("Prediction: This customer is likely to **stay**.")
st.write("---")
