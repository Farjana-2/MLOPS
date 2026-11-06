
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Select features
X = df[['Genre', 'Duration_Min',
        'Budget_Crore_INR', 'Popularity_Index']]

y = df['Rating_10']

# Preprocessing
preprocessor = ColumnTransformer([
    ('genre', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Genre']),

    ('numeric', SimpleImputer(strategy='median'),
     ['Duration_Min', 'Budget_Crore_INR',
      'Popularity_Index'])
])

# Create model
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "movie_model.pkl")

print("Model trained and saved successfully!")
