import joblib

model = joblib.load("AIML Journey\house-price-prediction\house_price_pipeline.pkl")

features = model["features"]
model = model["model"]

normal_sample = [[4,1.50,2480,6000,1,0,0,3,8,960,47.52,-122.39,1810,6000,71,0,3]]
luxury_sample = [[7,8,13540,307752,3,0,4,3,12,4130,47.67,-121.99,4850,217800,27,0,5]]
cheap_house = [[3,2.5,1670,5797,2,0,0,3,7,0,47.35,-122.18,1670,6183,38,0,7]]

bedroom = input("enter no of bedrooms = ")
bathrooms = input("enter no of bathrooms = ")
sqft_living = input("enter no of sqft_living = ")
sqft_lot = input("enter no of sqft_lot = ")
floors = input("enter no of floors = ")
waterfront = input("enter no of waterfront = ")
view = input("enter no of view = ")
condition = input("enter no of condition = ")
grade = input("enter no of grade = ")
sqft_basement = input("enter no of sqft_basement = ")
lat = input("enter no of lat = ")
long = input("enter no of long = ")
sqft_living15 = input("enter no of sqft_living15 = ")
sqft_lot15 = input("enter no of sqft_lot15 = ")
house_age = input("enter no of house_age = ")
renovated = input("enter no of renovated = ")
sale_month = input("enter no of sale_month = ")

sample = [[bedroom, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
 view, condition, grade, sqft_basement, lat, long, sqft_living15, sqft_lot15, house_age, renovated, sale_month]]

prediction = model.predict(sample)

print(prediction)