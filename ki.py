import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Maitama District ML App", layout="wide")

st.title("🚍 Maitama District Route Analysis & Speed Prediction (PDF Export Enabled)")

# --- Step 1: Dataset ---
data = {
    "ROUTE": [
        "Banex - Hospital Junction", 
        "Banex - University Junction", 
        "Banex - Wuse Market Junction",
        "Banex - Head of Service Junction", 
        "Hospital Junction - University Junction",
        "Hospital Junction - Wuse Market Junction", 
        "Hospital Junction - Head of Service",
        "Wuse Market Junction - University Junction", 
        "Wuse Market Junction - Head of Service Junction",
        "University Junction - Head of Service"
    ],
    "LENGTH_km": [2.5, 3.9, 1.7, 7.0, 1.3, 2.5, 4.8, 3.6, 5.3, 1.0],
    "TIME_sec": [471, 364, 101, 408, 132, 227, 218, 185, 312, 149],
    "AVG_SPEED": [19, 29, 62, 61, 35, 39, 28, 37, 62, 25]
}
df = pd.DataFrame(data)

st.subheader("📋 Maitama District Dataset")
st.dataframe(df)

# --- Step 2: Visualizations ---
st.subheader("📊 Visualization")

x_var = st.selectbox("X-axis variable:", ["LENGTH_km", "TIME_sec", "AVG_SPEED"])
y_var = st.selectbox("Y-axis variable:", ["LENGTH_km", "TIME_sec", "AVG_SPEED"])
plot_type = st.radio("Select Plot Type:", ["Scatter Plot", "Regression Plot"])

fig1, ax1 = plt.subplots()
if plot_type == "Scatter Plot":
    sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax1, s=100)
else:
    sns.regplot(x=x_var, y=y_var, data=df, ax=ax1)
ax1.set_title(f"{plot_type}: {y_var} vs {x_var}")
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots()
numeric_df = df[["LENGTH_km", "TIME_sec", "AVG_SPEED"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)

# --- Step 3: Train ML Model ---
X = df[["LENGTH_km", "TIME_sec"]]
y = df["AVG_SPEED"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("🧮 Model Performance")
st.metric("R² Score", f"{r2:.2f}")
st.metric("Mean Absolute Error", f"{mae:.2f}")

# --- Step 4: Prediction ---
st.subheader("🔮 Predict Average Speed")
length = st.number_input("Enter Route Length (km):", 0.0, 10.0, step=0.1)
time = st.number_input("Enter Time (sec):", 0, 1000, step=10)

if "predictions" not in st.session_state:
    st.session_state.predictions = []

if st.button("Predict Speed"):
    pred_speed = model.predict([[length, time]])[0]
    st.success(f"Predicted Speed: {pred_speed:.2f} km/h")
    st.session_state.predictions.append(
        {"Length (km)": length, "Time (sec)": time, "Predicted Speed (km/h)": round(pred_speed, 2)}
    )

if len(st.session_state.predictions) > 0:
    st.write("🧾 Recent Predictions")
    pred_df = pd.DataFrame(st.session_state.predictions)
    st.dataframe(pred_df)

# --- Step 5: Generate PDF Report ---
def generate_pdf(df, r2, mae, fig1, fig2, predictions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Maitama District Route Analysis Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Model Performance:")
    c.drawString(70, height - 100, f"R² Score: {r2:.2f}")
    c.drawString(70, height - 115, f"Mean Absolute Error: {mae:.2f}")

    # Save plots as images
    img1 = BytesIO()
    fig1.savefig(img1, format="png", bbox_inches="tight")
    img1.seek(0)
    img2 = BytesIO()
    fig2.savefig(img2, format="png", bbox_inches="tight")
    img2.seek(0)

    # Add plots
    c.drawImage(ImageReader(img1), 50, height - 420, width=250, height=200)
    c.drawImage(ImageReader(img2), 320, height - 420, width=250, height=200)

    c.drawString(50, height - 440, "Recent Predictions:")
    y_pos = height - 460
    c.setFont("Helvetica", 9)
    for i, row in enumerate(predictions):
        c.drawString(60, y_pos, f"{i+1}. L={row['Length (km)']} km, T={row['Time (sec)']} sec → {row['Predicted Speed (km/h)']} km/h")
        y_pos -= 15

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if st.button("📄 Generate PDF Report"):
    pdf = generate_pdf(df, r2, mae, fig1, fig2, st.session_state.predictions)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf,
        file_name="Maitama_District_Report.pdf",
        mime="application/pdf"
    )
