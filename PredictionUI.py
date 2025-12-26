import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="Earthquake Impact Predictor", page_icon="🌍", layout="wide")

@st.cache_resource
def load_artifacts():
    model = pickle.load(open("gb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, le

model, scaler, le = load_artifacts()

# ---------- Load dataset ----------
# Replace with your actual dataset path
df=pd.read_csv('C:/Users/Harshit Srivastava/OneDrive/Desktop/Internship/earthquake_alert_balanced_dataset.csv')

def alert_color(alert):
    return {
        "green": "#22c55e",
        "yellow": "#eab308",
        "orange": "#f97316",
        "red": "#ef4444",
        "error": "#ef4444"
    }.get(alert, "#6b7280")

# ---------- Header ----------
st.markdown(
    """
    <div style="text-align:center; margin-bottom:2rem;">
        <h1 style="font-size:3rem; font-weight:bold; background: -webkit-linear-gradient(#06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Earthquake Impact Predictor
        </h1>
        <p style="color:#6b7280; font-size:1.1rem;">
            Machine learning-powered alert level prediction based on seismic parameters.<br>
            Model trained on 1,300 earthquake records with 95.24% accuracy.
        </p>
        <span style="color:#3b82f6; font-size:0.9rem;">🔵 Real-time prediction engine</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Presets ----------
presets = [
    ("Minor Event", dict(magnitude=6.8, depth=115, cdi=4.4, mmi=5.2, sig=-39)),
    ("Moderate Event", dict(magnitude=7.2, depth=30, cdi=7, mmi=7, sig=50)),
    ("Significant Event", dict(magnitude=7.08, depth=40.8, cdi=7.3, mmi=6.9, sig=-23)),
    ("Major Event", dict(magnitude=7.8, depth=15, cdi=9, mmi=9, sig=100)),
]

def set_preset(vals):
    st.session_state.update(vals)

# ---------- Layout ----------
col_input, col_result = st.columns([1.2, 1])

with col_input:
    st.subheader("Seismic Parameters")

    mode = st.radio("Choose input mode:", ["Sliders", "Number Inputs"], horizontal=True)

    if mode == "Sliders":
        magnitude = st.slider("Magnitude", 0.0, 10.0, float(st.session_state.get("magnitude", 7.0)), 0.1, key="magnitude")
        depth = st.slider("Depth (km)", 0, 700, int(st.session_state.get("depth", 50)), 1, key="depth")
        cdi = st.slider("CDI", 1.0, 10.0, float(st.session_state.get("cdi", 6.0)), 0.1, key="cdi")
        mmi = st.slider("MMI", 1.0, 10.0, float(st.session_state.get("mmi", 6.0)), 0.1, key="mmi")
        sig = st.slider("SIG", -100, 200, int(st.session_state.get("sig", 0)), 1, key="sig")
    else:
        magnitude = st.number_input("Magnitude", 0.0, 10.0, float(st.session_state.get("magnitude", 7.0)), 0.1, key="magnitude")
        depth = st.number_input("Depth (km)", 0, 700, int(st.session_state.get("depth", 50)), 1, key="depth")
        cdi = st.number_input("CDI", 1.0, 10.0, float(st.session_state.get("cdi", 6.0)), 0.1, key="cdi")
        mmi = st.number_input("MMI", 1.0, 10.0, float(st.session_state.get("mmi", 6.0)), 0.1, key="mmi")
        sig = st.number_input("SIG", -100, 200, int(st.session_state.get("sig", 0)), 1, key="sig")

    st.markdown("#### Test Scenarios")
    preset_cols = st.columns(len(presets))
    for i, (label, vals) in enumerate(presets):
        with preset_cols[i]:
            st.button(label, use_container_width=True, on_click=set_preset, args=(vals,))

with col_result:
    st.subheader("Prediction Result")

    if st.button("Predict", use_container_width=True):
        sample = np.array([[st.session_state["magnitude"],
                            st.session_state["depth"],
                            st.session_state["cdi"],
                            st.session_state["mmi"],
                            st.session_state["sig"]]])
        sample_scaled = scaler.transform(sample)

        # Prediction
        pred = model.predict(sample_scaled)
        zone = le.inverse_transform(pred)[0]

        # Confidence score
        probs = model.predict_proba(sample_scaled)[0]
        pred_confidence = probs[pred[0]]

        bg = alert_color(zone)
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background-color:{bg}20;border:2px solid {bg}; text-align:center;">
                <h2 style="margin:0;color:{bg};">Predicted Alert: {zone.capitalize()}</h2>
                <p style="margin-top:8px;color:#374151;">
                    Magnitude {st.session_state["magnitude"]}, Depth {st.session_state["depth"]} km,
                    CDI {st.session_state["cdi"]}, MMI {st.session_state["mmi"]}, SIG {st.session_state["sig"]}
                </p>
                <p style="margin-top:8px;color:#374151;font-weight:bold;">
                    Confidence: {pred_confidence:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show dataset below prediction
        
        
        st.write("Preview of dataset (first 7 rows):")
        st.dataframe(df.head(7))
        with st.expander("See full dataset"):
            st.dataframe(df)
        

# ---------- Footer ----------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#6b7280; font-size:0.9rem;">
        Based on Gradient Boost classifier model. For educational purposes only.<br>
        Data: USGS Earthquake Catalog | Model accuracy: 95.24%
    </div>
    """,
    unsafe_allow_html=True
)