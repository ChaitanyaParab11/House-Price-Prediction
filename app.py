import streamlit as st
import joblib 

st.title("House Price Predictor")

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)
sqft_living = st.number_input("Area of Living Room (in sqft)", min_value=100)
sqft_lot = st.number_input("Area of Lot (in sqft)", min_value=100)
floors = st.number_input("Floors", min_value=1)
waterfront = st.number_input("Waterfront", min_value=0, max_value=1)
view = st.number_input("View", min_value=0, max_value=4)
condition = st.number_input("Condition", min_value=1, max_value=5)
grade = st.number_input("Grade", min_value=1, max_value=13)
sqft_basement = st.number_input("Area of Basement (in sqft)", min_value=100)
lat = st.number_input("Latitude")
long = st.number_input("Longtitude")
sqft_living15 = st.number_input("Area of 15 nearest Living Room (in sqft)", min_value=100)
sqft_lot15 = st.number_input("Area of 15 nearest Lot (in sqft)", min_value=100)
house_age = st.number_input("House Age", min_value=1)
renovated = st.number_input("Is it Renovated")
sale_month = st.number_input("Month of Sale", min_value=1, max_value=12)

model = joblib.load("house_price_pipeline.pkl")
model = model["model"]

if st.button("Predict Price"):
    input = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_basement, lat, long, sqft_living15, sqft_lot15, house_age, renovated, sale_month]]
    prediction = model.predict(input)

    st.success(f"Predicted Price : {prediction[0]}")