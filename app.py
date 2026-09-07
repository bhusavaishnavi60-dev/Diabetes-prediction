import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(to right,#dbeafe,#f0f9ff);
}

.main-title{
    text-align:center;
    color:#0f172a;
    font-size:40px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#2563eb;
    font-size:18px;
    margin-bottom:25px;
}

.login-box{
    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
    max-width:500px;
    margin:auto;
}

.card{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    margin-bottom:15px;
}

.result-positive{
    background:#fee2e2;
    padding:20px;
    border-radius:15px;
    border-left:8px solid red;
}

.result-negative{
    background:#dcfce7;
    padding:20px;
    border-radius:15px;
    border-left:8px solid green;
}

.tip{
    background:#eff6ff;
    padding:12px;
    border-radius:10px;
    margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "1234"
    }

if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI & Machine Learning Based Healthcare Prediction</p>", unsafe_allow_html=True)

    menu = st.radio("Choose Option", ["Login", "Signup"], horizontal=True)

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    if menu == "Login":
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

    else:
        st.subheader("👤 Create New Account")

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up", use_container_width=True):
            if new_pass != confirm:
                st.error("Passwords do not match!")
            elif new_user in st.session_state.users:
                st.warning("Username already exists!")
            else:
                st.session_state.users[new_user] = new_pass
                st.success("Account Created Successfully! Please Login.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
smoking_encoder = joblib.load("smoking_encoder.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.success(f"👋 Welcome {st.session_state.username}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ---------------- HEADER ----------------
st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>AI & Machine Learning Based Healthcare Prediction</p>", unsafe_allow_html=True)

st.image(
    "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200",
    use_container_width=True
)

st.write("")

# ---------------- INPUT FORM ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📝 Enter Patient Health Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    age = st.slider("🎂 Age", 1, 100, 25)
    hypertension = st.selectbox("💓 Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("❤️ Heart Disease", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox(
        "🚬 Smoking History",
        ["never", "current", "former", "No Info", "ever", "not current"]
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0
    )

    hba1c = st.number_input(
        "🩸 HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    blood_glucose = st.number_input(
        "🧪 Blood Glucose Level",
        min_value=50,
        max_value=300,
        value=100
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Diabetes", use_container_width=True):

    gender_value = gender_encoder.transform([gender])[0]
    smoking_value = smoking_encoder.transform([smoking_history])[0]

    patient = np.array([[gender_value,
                         age,
                         1 if hypertension == "Yes" else 0,
                         1 if heart_disease == "Yes" else 0,
                         smoking_value,
                         bmi,
                         hba1c,
                         blood_glucose]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    st.write("")
    st.subheader("📊 Prediction Result")

    st.progress(int(probability))
    st.metric("Prediction Confidence", f"{probability:.2f}%")

    if prediction == 1:

        st.markdown("""
        <div class='result-positive'>
        <h2>🔴 Positive Diabetes Prediction</h2>
        <p>The model predicts a higher risk of diabetes.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🩺 Recommended Precautions")

        precautions = [
            "🥗 Eat a balanced low-sugar diet.",
            "🏃 Exercise at least 30 minutes every day.",
            "💧 Drink plenty of water.",
            "🩸 Monitor blood sugar regularly.",
            "⚖️ Maintain a healthy body weight.",
            "🚭 Avoid smoking and alcohol.",
            "👨‍⚕️ Consult a doctor for confirmation."
        ]

        for tip in precautions:
            st.markdown(f"<div class='tip'>{tip}</div>", unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result-negative'>
        <h2>🟢 Negative Diabetes Prediction</h2>
        <p>The model predicts a lower risk of diabetes.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💚 Healthy Lifestyle Tips")

        tips = [
            "🥗 Maintain a healthy balanced diet.",
            "🏃 Exercise regularly.",
            "💧 Drink enough water every day.",
            "😴 Sleep for 7–8 hours daily.",
            "🍎 Reduce sugar intake.",
            "🩺 Get regular health checkups."
        ]

        for tip in tips:
            st.markdown(f"<div class='tip'>{tip}</div>", unsafe_allow_html=True)

