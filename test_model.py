
import joblib
import pandas as pd
import numpy as np

def test_movie_prediction():

    model = joblib.load("movie_model.pkl")

    movie = pd.DataFrame([{
        'Genre': 'Action',
        'Duration_Min': 120,
        'Budget_Crore_INR': 50,
        'Popularity_Index': 80
    }])

    result = model.predict(movie)

    assert len(result) == 1
    assert np.isfinite(result[0])

    print("Movie prediction test passed!")
