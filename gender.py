import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)
# Load and preprocess data
df = pd.read_csv("dataset.csv")
df.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # Male=1, Female=0
df['smoking_status'] = le.fit_transform(df['smoking_status'])

features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
scaler = StandardScaler()

# Train model for each gender
for gender, label in zip(['female', 'male'], [0, 1]):
    gender_df = df[df['gender'] == label]
    X = gender_df[features]
    y = gender_df['stroke']
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{gender}_model.pkl")

    print(f"{gender.capitalize()} model trained and saved.")
