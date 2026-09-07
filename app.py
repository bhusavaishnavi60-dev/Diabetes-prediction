from flask import Flask, render_template, request, redirect, session
import sqlite3
import hashlib
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "diabetes_prediction_secret"

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
smoking_encoder = joblib.load("smoking_encoder.pkl")

# SQLite Database
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

# Password Encryption
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- LOGIN PAGE ----------------
@app.route("/")
def index():
    return redirect("/login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = hash_password(request.form["password"])

        try:
            cursor.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (username, password)
            )
            conn.commit()
            return redirect("/login")
        except:
            return "Username already exists!"

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = hash_password(request.form["password"])

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()

        if user:
            session["username"] = username
            return redirect("/home")

        return "Invalid Username or Password"

    return render_template("login.html")


# ---------------- HOME PAGE ----------------
@app.route("/home")
def home():
    if "username" not in session:
        return redirect("/login")

    return render_template("home.html", username=session["username"])


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    gender = gender_encoder.transform([request.form["gender"]])[0]
    smoking = smoking_encoder.transform([request.form["smoking_history"]])[0]

    patient = np.array([[
        gender,
        float(request.form["age"]),
        int(request.form["hypertension"]),
        int(request.form["heart_disease"]),
        smoking,
        float(request.form["bmi"]),
        float(request.form["HbA1c_level"]),
        float(request.form["blood_glucose_level"])
    ]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    # Positive Prediction
    if prediction == 1:
        result = "🔴 Positive Diabetes Prediction"

        precautions = [
            "Eat a balanced low-sugar diet.",
            "Exercise for at least 30 minutes daily.",
            "Drink plenty of water every day.",
            "Monitor blood sugar regularly.",
            "Maintain a healthy body weight.",
            "Avoid smoking and alcohol.",
            "Consult a doctor for confirmation."
        ]

    # Negative Prediction
    else:
        result = "🟢 Negative Diabetes Prediction"

        precautions = [
            "Maintain a healthy balanced diet.",
            "Exercise regularly.",
            "Drink enough water.",
            "Sleep for 7–8 hours daily.",
            "Reduce sugar intake.",
            "Get regular health checkups."
        ]

    return render_template(
        "result.html",
        result=result,
        probability=round(probability, 2),
        precautions=precautions
    )


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)