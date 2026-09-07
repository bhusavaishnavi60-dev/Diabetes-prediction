import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Read dataset
data = pd.read_csv("diabetes/diabetes_prediction_dataset.csv")

# Convert text to numbers
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

data["gender"] = gender_encoder.fit_transform(data["gender"])
data["smoking_history"] = smoking_encoder.fit_transform(data["smoking_history"])

# Save encoders
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(smoking_encoder, "smoking_encoder.pkl")

# Inputs and output
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Accuracy:", model.score(X_test, y_test))
print("Model Saved Successfully!")