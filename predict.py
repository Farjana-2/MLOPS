
import joblib
import pandas as pd

# Load model
model = joblib.load("movie_model.pkl")

# Movie details
movie = pd.DataFrame([{
    'Genre': 'Action',
    'Duration_Min': 120,
    'Budget_Crore_INR': 50,
    'Popularity_Index': 80
}])

# Predict rating
prediction = model.predict(movie)

print("Predicted Movie Rating:", round(prediction[0], 2))
