import time
import json
import pickle
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(page_title="Virtual ICU AI Monitor", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
    .title-big {font-size: 2.25rem; font-weight: 900; margin: 0.2rem 0 0.6rem 0;}
    .subtle {opacity: 0.85;}
    .card {padding: 1rem; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);}
    .pill {display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10);}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CLINICAL SCORING (simplified)
# =========================================================
def news2_score(temp, rr, spo2, sbp, hr, consciousness="Alert", supplemental_o2=False):
    score = 0
    if temp <= 35.0: score += 3
    elif temp <= 36.0: score += 1
    elif temp <= 38.0: score += 0
    elif temp <= 39.0: score += 1
    else: score += 2

    if rr <= 8: score += 3
    elif rr <= 11: score += 1
    elif rr <= 20: score += 0
    elif rr <= 24: score += 2
    else: score += 3

    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 94: score += 1
    else: score += 0

    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    else: score += 0

    if hr <= 40: score += 3
    elif hr <= 50: score += 1
    elif hr <= 90: score += 0
    elif hr <= 110: score += 1
    elif hr <= 130: score += 2
    else: score += 3

    if consciousness != "Alert":
        score += 3
    if supplemental_o2:
        score += 2

    return int(score)

def qsofa_score(rr, sbp, altered_mentation=False):
    score = 0
    if rr >= 22: score += 1
    if sbp <= 100: score += 1
    if altered_mentation: score += 1
    return int(score)

def shock_index(hr, sbp):
    return float(hr) / float(sbp) if sbp else 0.0

def risk_tier_from_news2(score):
    if score <= 4: return "Low"
    if score <= 6: return "Medium"
    return "High"

# =========================================================
# GAUGES
# =========================================================
def gauge(title, value, unit, vmin, vmax, steps):
    fig = go.Figure(
        data=[go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": f" {unit}"},
            gauge={"axis": {"range": [vmin, vmax]}, "steps": steps},
        )]
    )
    fig.update_layout(height=220, margin=dict(l=8, r=8, t=10, b=8), template="plotly_dark")
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, use_container_width=True)

def news2_risk_meter(news2_val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(news2_val),
        title={"text": "NEWS2 Risk Meter"},
        gauge={
            "axis": {"range": [0, 20]},
            "steps": [
                {"range": [0, 4], "color": "#51CF66"},
                {"range": [5, 6], "color": "#FFA500"},
                {"range": [7, 20], "color": "#FF6B6B"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": 7},
        },
    ))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# =========================================================
# MODEL MANAGER (switch ML model)
# =========================================================
def _try_load_json_file(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _try_load_pickle_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_local_assets(model_path: str, scaler_path: str, feature_names_path: str, metrics_path: str):
    model = _try_load_pickle_file(model_path)
    scaler = _try_load_pickle_file(scaler_path) if scaler_path else None
    feature_names = _try_load_json_file(feature_names_path) if feature_names_path else None
    metrics = _try_load_json_file(metrics_path) if metrics_path else None
    return model, scaler, feature_names, metrics

def load_uploaded_pickle(uploaded_file):
    # Only load trusted uploads (pickle can execute code) [web:144]
    return pickle.loads(uploaded_file.read())  # [web:144]

def ml_predict(model, scaler, feature_names, row_dict):
    X = np.array([float(row_dict.get(feat, 0)) for feat in feature_names], dtype=float).reshape(1, -1)
    Xs = scaler.transform(X) if scaler is not None else X
    pred = int(model.predict(Xs)[0])
    prob_high = float(model.predict_proba(Xs)[0, 1]) if hasattr(model, "predict_proba") else float(pred)
    return pred, prob_high

def ensure_model_in_state():
    if "ml_model" not in st.session_state:
        st.session_state.ml_model = None
    if "ml_scaler" not in st.session_state:
        st.session_state.ml_scaler = None
    if "ml_feature_names" not in st.session_state:
        st.session_state.ml_feature_names = None
    if "ml_metrics" not in st.session_state:
        st.session_state.ml_metrics = None
    if "ml_source" not in st.session_state:
        st.session_state.ml_source = "LOCAL:gb_model.pkl"

ensure_model_in_state()

# =========================================================
# SIMULATION ENGINE
# =========================================================
def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def make_patient(patient_id, profile):
    if profile == "stable":
        vitals = dict(temp=36.9, hr=78, rr=14, spo2=97, sbp=128)
        drift = dict(temp=0.000, hr=0.00, rr=0.00, spo2=0.00, sbp=0.00)
    elif profile == "developing_sepsis":
        vitals = dict(temp=38.3, hr=105, rr=22, spo2=94, sbp=105)
        drift = dict(temp=+0.010, hr=+0.06, rr=+0.04, spo2=-0.01, sbp=-0.04)
    elif profile == "cardiac_risk":
        vitals = dict(temp=37.2, hr=115, rr=18, spo2=95, sbp=100)
        drift = dict(temp=+0.000, hr=+0.05, rr=+0.02, spo2=-0.005, sbp=-0.05)
    else:  # respiratory_decline
        vitals = dict(temp=37.1, hr=92, rr=24, spo2=92, sbp=120)
        drift = dict(temp=+0.002, hr=+0.03, rr=+0.05, spo2=-0.02, sbp=-0.01)

    return {
        "patient_id": patient_id,
        "profile": profile,
        "age": int(np.random.randint(18, 86)),
        "altered_mentation": False,
        "supp_o2": False,
        "vitals": vitals,
        "drift": drift,
    }

def init_sim(n_patients=10):
    profiles = np.random.choice(
        ["stable", "developing_sepsis", "cardiac_risk", "respiratory_decline"],
        size=n_patients,
        p=[0.55, 0.2, 0.15, 0.10],
    )
    pts = []
    for i in range(n_patients):
        pts.append(make_patient(f"P{(i+1):03d}", str(profiles[i])))
    return pts

def step_sim(patients, noise=0.12):
    for p in patients:
        v = p["vitals"]
        d = p["drift"]

        v["temp"] = clamp(v["temp"] + d["temp"] + np.random.normal(0, noise*0.10), 35.0, 41.0)
        v["hr"]   = clamp(v["hr"]   + d["hr"]   + np.random.normal(0, noise*2.0), 40, 160)
        v["rr"]   = clamp(v["rr"]   + d["rr"]   + np.random.normal(0, noise*0.9), 8, 40)
        v["spo2"] = clamp(v["spo2"] + d["spo2"] + np.random.normal(0, noise*0.6), 70, 100)
        v["sbp"]  = clamp(v["sbp"]  + d["sbp"]  + np.random.normal(0, noise*2.0), 70, 180)

        if p["profile"] != "stable" and np.random.rand() < 0.01:
            p["altered_mentation"] = not p["altered_mentation"]

    return patients

def sim_df(patients, sim_minute, start_ts):
    rows = []
    ts = f"{start_ts} +{sim_minute}m"
    for p in patients:
        v = p["vitals"]
        n2 = news2_score(
            v["temp"], v["rr"], v["spo2"], v["sbp"], v["hr"],
            "Confused" if p["altered_mentation"] else "Alert",
            p["supp_o2"]
        )
        qf = qsofa_score(v["rr"], v["sbp"], p["altered_mentation"])
        si = shock_index(v["hr"], v["sbp"])
        rows.append({
            "timestamp": ts,
            "sim_minute": int(sim_minute),
            "patient_id": p["patient_id"],
            "profile": p["profile"],
            "age": p["age"],
            "temp": round(v["temp"], 2),
            "hr": int(round(v["hr"])),
            "rr": int(round(v["rr"])),
            "spo2": round(v["spo2"], 1),
            "sbp": int(round(v["sbp"])),
            "supp_o2": bool(p["supp_o2"]),
            "altered_mentation": bool(p["altered_mentation"]),
            "news2": int(n2),
            "qsofa": int(qf),
            "shock_index": round(si, 3),
            "risk_tier": risk_tier_from_news2(n2),
        })
    df = pd.DataFrame(rows)
    tier_order = pd.Categorical(df["risk_tier"], categories=["High", "Medium", "Low"], ordered=True)
    df["tier_order"] = tier_order
    return df.sort_values(["tier_order", "news2"], ascending=[True, False]).drop(columns=["tier_order"])

def apply_invigilator_edits(patients, edited_df):
    mp = {p["patient_id"]: p for p in patients}
    for _, row in edited_df.iterrows():
        pid = row["patient_id"]
        if pid not in mp:
            continue
        p = mp[pid]
        p["profile"] = str(row.get("profile", p["profile"]))
        p["age"] = int(row.get("age", p["age"]))
        p["supp_o2"] = bool(row.get("supp_o2", p["supp_o2"]))
        p["altered_mentation"] = bool(row.get("altered_mentation", p["altered_mentation"]))
        p["vitals"]["temp"] = float(row.get("temp", p["vitals"]["temp"]))
        p["vitals"]["hr"]   = float(row.get("hr", p["vitals"]["hr"]))
        p["vitals"]["rr"]   = float(row.get("rr", p["vitals"]["rr"]))
        p["vitals"]["spo2"] = float(row.get("spo2", p["vitals"]["spo2"]))
        p["vitals"]["sbp"]  = float(row.get("sbp", p["vitals"]["sbp"]))
    return patients

# =========================================================
# SESSION STATE
# =========================================================
if "mode" not in st.session_state:
    st.session_state.mode = "SIMULATION"

if "sim_patients" not in st.session_state:
    st.session_state.sim_patients = init_sim(n_patients=10)

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

if "sim_tick_ms" not in st.session_state:
    st.session_state.sim_tick_ms = 1000

if "sim_noise" not in st.session_state:
    st.session_state.sim_noise = 0.12

if "sim_history" not in st.session_state:
    st.session_state.sim_history = {}

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = "P001"

if "sim_total_minutes" not in st.session_state:
    st.session_state.sim_total_minutes = 480

if "sim_step_minutes" not in st.session_state:
    st.session_state.sim_step_minutes = 3

if "sim_position" not in st.session_state:
    st.session_state.sim_position = 0

if "sim_start_time" not in st.session_state:
    st.session_state.sim_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.markdown("## 🏥 Virtual ICU AI Monitor")
    st.markdown("<div class='subtle'>Disease Prediction & Early Warning System</div>", unsafe_allow_html=True)
    st.divider()

    mode_label = st.selectbox(
        "Data Source",
        ["SIMULATION (live patients)", "REAL DATASET (CSV + ML)"],
        index=0 if st.session_state.mode == "SIMULATION" else 1
    )
    st.session_state.mode = "SIMULATION" if "SIMULATION" in mode_label else "REAL_DATASET"

    selected_page = option_menu(
        "Navigation",
        ["Dashboard", "Patient Monitor", "Invigilator", "Model Performance", "Model Manager"],
        icons=["speedometer2", "heart-pulse", "person-gear", "bar-chart", "cpu"],
        default_index=0,
    )

    st.divider()
    st.markdown("### Monitoring Parameters")
    analysis_window_min = st.slider("Analysis Window (minutes)", 5, 60, 15, step=5)

    if st.session_state.mode == "SIMULATION":
        st.markdown("### Simulation Timeline")
        st.session_state.sim_total_minutes = st.slider("Simulation Length (minutes)", 60, 1440, int(st.session_state.sim_total_minutes), step=30)
        st.session_state.sim_step_minutes = st.slider("Simulation Speed (minutes/refresh)", 1, 15, int(st.session_state.sim_step_minutes), step=1)
        st.session_state.sim_tick_ms = st.slider("Refresh Rate (ms)", 300, 3000, int(st.session_state.sim_tick_ms), step=100)
        st.session_state.sim_noise = st.slider("Noise level", 0.0, 0.5, float(st.session_state.sim_noise), step=0.01)
        st.session_state.sim_position = st.slider("Manual Position (minutes from start)", 0, int(st.session_state.sim_total_minutes), int(st.session_state.sim_position), step=1)

        st.markdown("### Simulation Controls")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶", use_container_width=True):
                st.session_state.sim_running = True
        with c2:
            if st.button("⏸", use_container_width=True):
                st.session_state.sim_running = False
        with c3:
            if st.button("⟳", use_container_width=True):
                st.session_state.sim_patients = init_sim(n_patients=10)
                st.session_state.sim_history = {}
                st.session_state.selected_patient = "P001"
                st.session_state.sim_position = 0
                st.session_state.sim_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.sim_running = False
                st.rerun()

# =========================================================
# LOAD REAL DATASET FILES (REAL_DATASET mode)
# =========================================================
errors = []
engineered = None
feature_importance_df = None

if st.session_state.mode == "REAL_DATASET":
    try:
        engineered = pd.read_csv("engineered_features.csv")
    except Exception as e:
        errors.append(f"engineered_features.csv not loaded: {e}")

    try:
        feature_importance_df = pd.read_csv("feature_importance.csv")
    except Exception as e:
        errors.append(f"feature_importance.csv not loaded: {e}")

# =========================================================
# SIMULATION: current frame + history
# =========================================================
if st.session_state.mode == "SIMULATION":
    df_now = sim_df(
        st.session_state.sim_patients,
        sim_minute=int(st.session_state.sim_position),
        start_ts=st.session_state.sim_start_time
    )

    for _, r in df_now.iterrows():
        pid = r["patient_id"]
        st.session_state.sim_history.setdefault(pid, [])
        st.session_state.sim_history[pid].append(dict(r))
        if len(st.session_state.sim_history[pid]) > 400:
            st.session_state.sim_history[pid] = st.session_state.sim_history[pid][-400:]

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='title-big'>🏥 Virtual ICU Monitor - AI Disease Prediction System</div>", unsafe_allow_html=True)

if errors:
    st.error("Fix these errors first:")
    for e in errors:
        st.write(f"- {e}")

# =========================================================
# PAGES
# =========================================================
if selected_page == "Model Manager":
    st.subheader("🧠 Model Manager (Change ML model)")
    st.caption("Load a different model for REAL DATASET predictions.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Use local model file")
        local_model = st.selectbox("Select model file", ["gb_model.pkl", "rf_model.pkl", "nn_model.pkl"], index=0)
        scaler_path = st.text_input("Scaler file (optional)", value="gb_scaler.pkl")
        feature_names_path = st.text_input("Feature names JSON", value="feature_names.json")
        metrics_path = st.text_input("Metrics JSON (optional)", value="model_metrics.json")

        if st.button("Load selected local model"):
            try:
                load_local_assets.clear()  # clear cached resource for reload behavior [web:50]
                model, scaler, f_names, metrics = load_local_assets(local_model, scaler_path, feature_names_path, metrics_path)

                st.session_state.ml_model = model
                st.session_state.ml_scaler = scaler
                st.session_state.ml_feature_names = f_names
                st.session_state.ml_metrics = metrics
                st.session_state.ml_source = f"LOCAL:{local_model}"
                st.success(f"Loaded local model: {local_model}")
            except Exception as e:
                st.error(f"Failed to load local model: {e}")

    with c2:
        st.markdown("### Upload your model")
        st.caption("Upload only models you trust (pickle safety).")
        up_model = st.file_uploader("Upload model (.pkl/.joblib)", type=["pkl", "pickle", "joblib"])
        up_scaler = st.file_uploader("Upload scaler (optional .pkl/.joblib)", type=["pkl", "pickle", "joblib"])
        up_features = st.file_uploader("Upload feature_names.json", type=["json"])
        up_metrics = st.file_uploader("Upload model_metrics.json (optional)", type=["json"])

        if st.button("Load uploaded model"):
            try:
                if up_model is None:
                    st.error("Upload a model first.")
                elif up_features is None:
                    st.error("Upload feature_names.json first (must match training).")
                else:
                    model = load_uploaded_pickle(up_model)  # [web:144]
                    scaler = load_uploaded_pickle(up_scaler) if up_scaler is not None else None  # [web:144]
                    f_names = json.loads(up_features.read().decode("utf-8"))
                    metrics = json.loads(up_metrics.read().decode("utf-8")) if up_metrics is not None else None

                    st.session_state.ml_model = model
                    st.session_state.ml_scaler = scaler
                    st.session_state.ml_feature_names = f_names
                    st.session_state.ml_metrics = metrics
                    st.session_state.ml_source = f"UPLOAD:{up_model.name}"
                    st.success("Loaded uploaded model into current session.")
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")

    st.divider()
    st.markdown("### Active model")
    st.write(f"Source: **{st.session_state.ml_source}**")
    st.write(f"Has scaler: **{st.session_state.ml_scaler is not None}**")
    st.write(f"Feature count: **{len(st.session_state.ml_feature_names) if st.session_state.ml_feature_names else 0}**")

elif selected_page == "Dashboard":
    if st.session_state.mode == "SIMULATION":
        progress = st.session_state.sim_position / max(1, st.session_state.sim_total_minutes)
        st.progress(progress, text=f"Simulation progress: {st.session_state.sim_position}/{st.session_state.sim_total_minutes} minutes")

        st.caption(
            f"Start: {st.session_state.sim_start_time} | "
            f"Status: {'▶ Running' if st.session_state.sim_running else '⏸ Paused'} | "
            f"Refresh: {st.session_state.sim_tick_ms}ms | Step: {st.session_state.sim_step_minutes} min/tick"
        )

        total_patients = int(df_now["patient_id"].nunique())
        high = int((df_now["risk_tier"] == "High").sum())
        med = int((df_now["risk_tier"] == "Medium").sum())
        low = int((df_now["risk_tier"] == "Low").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", total_patients)
        c2.metric("High risk", high)
        c3.metric("Medium risk", med)
        c4.metric("Low risk", low)

        st.divider()

        st.subheader("Available Patients")
        patient_labels = [f"{r.patient_id}: {r.profile} (age {r.age})" for r in df_now[["patient_id","profile","age"]].itertuples(index=False)]
        selected_idx = st.selectbox("Select Patient", range(len(patient_labels)), format_func=lambda i: patient_labels[i])
        st.session_state.selected_patient = df_now.iloc[int(selected_idx)]["patient_id"]
        st.write(f"**Currently monitoring:** {st.session_state.selected_patient}")

        st.subheader("Current Patient Table")
        st.dataframe(df_now, use_container_width=True, height=420)

        st.subheader("NEWS2 Distribution")
        fig = px.histogram(df_now, x="news2", color="risk_tier", nbins=12,
                           color_discrete_map={"Low":"#51CF66","Medium":"#FFA500","High":"#FF6B6B"})
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)

    else:
        if engineered is None or errors:
            st.info("Dataset mode needs engineered_features.csv.")
            st.stop()

        latest = engineered.sort_values("timestamp").groupby("patient_id").tail(1)
        total_patients = int(latest["patient_id"].nunique())
        high_risk = int((latest["is_high_risk"] == 1).sum())
        avg_news2 = float(latest["news2_score"].mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients", total_patients)
        c2.metric("High-risk (label)", high_risk)
        c3.metric("Avg NEWS2", f"{avg_news2:.2f}")

        st.write(f"Active ML model: **{st.session_state.ml_source}**")

elif selected_page == "Patient Monitor":
    if st.session_state.mode == "SIMULATION":
        pid = st.session_state.selected_patient
        current = df_now[df_now["patient_id"] == pid].iloc[0]

        st.caption(f"Patient {pid} | Profile: {current['profile']} | Time: {current['timestamp']}")
        st.markdown(f"<span class='pill'>Risk: {current['risk_tier']}</span>", unsafe_allow_html=True)

        g1, g2, g3, g4, g5 = st.columns(5)
        with g1:
            gauge("Heart Rate", current["hr"], "bpm", 40, 160,
                  [{"range":[40,60],"color":"#4facfe"},{"range":[60,100],"color":"#51CF66"},{"range":[100,130],"color":"#FFA500"},{"range":[130,160],"color":"#FF6B6B"}])
        with g2:
            gauge("Systolic BP", current["sbp"], "mmHg", 70, 180,
                  [{"range":[70,90],"color":"#FF6B6B"},{"range":[90,110],"color":"#FFA500"},{"range":[110,180],"color":"#51CF66"}])
        with g3:
            gauge("SpO₂", current["spo2"], "%", 70, 100,
                  [{"range":[70,90],"color":"#FF6B6B"},{"range":[90,94],"color":"#FFA500"},{"range":[94,100],"color":"#51CF66"}])
        with g4:
            gauge("Resp Rate", current["rr"], "/min", 8, 40,
                  [{"range":[8,12],"color":"#4facfe"},{"range":[12,20],"color":"#51CF66"},{"range":[20,24],"color":"#FFA500"},{"range":[24,40],"color":"#FF6B6B"}])
        with g5:
            gauge("Temperature", current["temp"], "°C", 35, 41,
                  [{"range":[35,36],"color":"#4facfe"},{"range":[36,37.5],"color":"#51CF66"},{"range":[37.5,39],"color":"#FFA500"},{"range":[39,41],"color":"#FF6B6B"}])

        st.divider()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Clinical Scores")
            st.write(f"NEWS2 Score: **{int(current['news2'])}**")
            st.write(f"qSOFA Score: **{int(current['qsofa'])}**")
            st.write(f"Shock Index: **{float(current['shock_index']):.3f}**")
            st.markdown("</div>", unsafe_allow_html=True)
            st.plotly_chart(news2_risk_meter(current["news2"]), use_container_width=True)

        with c2:
            hist = pd.DataFrame(st.session_state.sim_history.get(pid, []))
            if not hist.empty:
                fig = px.line(hist, x="sim_minute", y=["news2", "qsofa", "shock_index"], markers=True, title="Risk Trends (updates every tick)")
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)

            hist = pd.DataFrame(st.session_state.sim_history.get(pid, []))
            if not hist.empty:
                cA, cB = st.columns(2)
                with cA:
                    fig = px.line(hist, x="sim_minute", y="hr", markers=True, title="Heart Rate (bpm)")
                    fig.update_layout(template="plotly_dark", height=260)
                    st.plotly_chart(fig, use_container_width=True)
                with cB:
                    fig = px.line(hist, x="sim_minute", y="sbp", markers=True, title="Blood Pressure (mmHg)")
                    fig.update_layout(template="plotly_dark", height=260)
                    st.plotly_chart(fig, use_container_width=True)

                cC, cD = st.columns(2)
                with cC:
                    fig = px.line(hist, x="sim_minute", y="spo2", markers=True, title="Oxygen Saturation (%)")
                    fig.update_layout(template="plotly_dark", height=260)
                    st.plotly_chart(fig, use_container_width=True)
                with cD:
                    fig = px.line(hist, x="sim_minute", y="rr", markers=True, title="Respiratory Rate (/min)")
                    fig.update_layout(template="plotly_dark", height=260)
                    st.plotly_chart(fig, use_container_width=True)

                fig = px.line(hist, x="sim_minute", y="temp", markers=True, title="Temperature (°C)")
                fig.update_layout(template="plotly_dark", height=260)
                st.plotly_chart(fig, use_container_width=True)

    else:
        if engineered is None or errors:
            st.info("Dataset mode needs engineered_features.csv.")
            st.stop()

        patient_list = sorted(engineered["patient_id"].unique())
        pid = st.selectbox("Select patient", patient_list)

        recs = engineered[engineered["patient_id"] == pid].sort_values("timestamp")
        latest = recs.iloc[-1]

        st.caption(f"Patient {pid} | Latest: {latest['timestamp']}")
        st.write(f"Active ML model: **{st.session_state.ml_source}**")

        if st.session_state.ml_model is None or st.session_state.ml_feature_names is None:
            st.warning("No ML model loaded. Go to Model Manager and load one.")
        else:
            pred, prob_high = ml_predict(st.session_state.ml_model, st.session_state.ml_scaler, st.session_state.ml_feature_names, dict(latest))
            if pred == 1:
                st.error(f"ML: HIGH RISK (P(high)={prob_high*100:.2f}%)")
            else:
                st.success(f"ML: LOW RISK (P(high)={prob_high*100:.2f}%)")

elif selected_page == "Invigilator":
    if st.session_state.mode != "SIMULATION":
        st.info("Invigilator panel is available in SIMULATION mode.")
        st.stop()

    st.subheader("🧑‍🏫 Invigilator Controls (Live Edit)")
    editable_cols = ["patient_id", "profile", "age", "temp", "hr", "rr", "spo2", "sbp", "supp_o2", "altered_mentation"]
    edited = st.data_editor(df_now[editable_cols].copy(), use_container_width=True, hide_index=True, num_rows="fixed")  # [web:72]

    if st.button("✅ Apply changes"):
        st.session_state.sim_patients = apply_invigilator_edits(st.session_state.sim_patients, edited)
        st.success("Applied edits.")
        st.rerun()

else:  # Model Performance
    if st.session_state.mode != "REAL_DATASET":
        st.info("Model performance is available in REAL DATASET mode.")
        st.stop()

    st.subheader("📈 Model Performance")
    st.write(f"Active ML model: **{st.session_state.ml_source}**")

    # Load metrics (prefer from session_state, else model_metrics.json)
    metrics = st.session_state.get("ml_metrics", None)
    if metrics is None:
        try:
            with open("model_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    if metrics is None:
        st.warning("No model metrics found. Provide model_metrics.json or upload it in Model Manager.")
        st.stop()

    # choose which metrics to show (if you have multiple)
    model_choice = st.selectbox(
        "Which metrics to display?",
        ["gradient_boosting", "random_forest", "neural_network"],
        index=0
    )
    chosen = metrics.get(model_choice, {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{float(chosen.get('accuracy', 0))*100:.2f}%")     # [web:188]
    c2.metric("Precision", f"{float(chosen.get('precision', 0))*100:.2f}%")   # [web:188]
    c3.metric("Recall", f"{float(chosen.get('recall', 0))*100:.2f}%")         # [web:188]
    c4.metric("F1", f"{float(chosen.get('f1', 0))*100:.2f}%")                 # [web:188]
    c5.metric("AUC", f"{float(chosen.get('auc', 0)):.4f}")                    # [web:188]

    st.divider()

    if feature_importance_df is None:
        try:
            feature_importance_df = pd.read_csv("feature_importance.csv")
        except Exception:
            feature_importance_df = None

    if feature_importance_df is not None:
        st.subheader("Feature importance (Top 15)")
        top = feature_importance_df.head(15).copy()
        fig = px.bar(top, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AUTO-UPDATE LOOP (simulation updates every instance)
# =========================================================
if st.session_state.mode == "SIMULATION" and st.session_state.sim_running:
    st.session_state.sim_position = min(
        int(st.session_state.sim_total_minutes),
        int(st.session_state.sim_position) + int(st.session_state.sim_step_minutes),
    )
    if st.session_state.sim_position >= int(st.session_state.sim_total_minutes):
        st.session_state.sim_running = False

    st.session_state.sim_patients = step_sim(st.session_state.sim_patients, noise=float(st.session_state.sim_noise))
    time.sleep(int(st.session_state.sim_tick_ms) / 1000.0)
    st.rerun()  # rerun updates UI immediately [web:126]
