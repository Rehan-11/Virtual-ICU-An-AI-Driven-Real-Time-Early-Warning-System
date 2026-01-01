# Virtual ICU (vICU) System — Complete Documentation Index

Welcome to the Virtual ICU AI Monitor project! This is your navigation guide.

---

## 📚 Documentation Files (Read in Order)

### 1. **INDEX.md** (You are here)
Quick overview, file structure, and navigation.

### 2. **QUICKSTART.md** 
5–10 minute setup guide + common errors + verification checklist.

### 3. **README.md**
Full technical documentation with features, modes, architecture, and customization.

### 4. **REQUIREMENTS.md**
All Python dependencies explained (required vs optional).

---

## 🚀 Start Here (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run streamlit_app.py

# 3. Open browser
# → http://localhost:8501 (auto-opens)
```

---

## 📂 Project Structure

```
project_root/
│
├── streamlit_app.py              ← Main application (full-featured)
├── requirements.txt              ← Python dependencies
│
├── INDEX.md                       ← This file (navigation)
├── QUICKSTART.md                  ← 5-min setup guide
├── README.md                      ← Full documentation
├── REQUIREMENTS.md                ← Dependency details
│
├── Data Files (SIMULATION mode)
│   └── (no files needed—generates synthetic patients)
│
├── Data Files (REAL DATASET mode - optional)
│   ├── engineered_features.csv    ← Patient records with clinical scores
│   ├── feature_names.json         ← ML feature list (for inference)
│   ├── feature_importance.csv     ← Feature importance for visualization
│   └── model_metrics.json         ← Model performance metrics
│
└── ML Models (optional, via Model Manager)
    ├── gb_model.pkl              ← Gradient Boosting model
    ├── rf_model.pkl              ← Random Forest model
    ├── nn_model.pkl              ← Neural Network model
    ├── gb_scaler.pkl             ← Feature scaler (optional)
    └── (or upload your own via UI)
```

---

## 🎯 What This App Does

### Simulation Mode (Default)
✅ **Live multi-patient ICU simulation** with realistic vital sign drift  
✅ **Real-time charts** that update every tick (configurable 300–3000ms)  
✅ **Invigilator panel** to manually edit patient vitals live (data editor)  
✅ **Timeline control**: seek to any minute, adjust simulation speed  
✅ **Clinical scoring**: NEWS2 (early warning), qSOFA, Shock Index  

### Real Dataset Mode
✅ **Load CSV data** (`engineered_features.csv`) with real patient records  
✅ **Patient monitor** with gauges + trend charts  
✅ **ML predictions** using loaded model + optional scaler  

### Model Manager
✅ **Switch ML models** (gradient boosting, random forest, neural network)  
✅ **Upload custom models** (.pkl files) with feature list  
✅ **Manage scalers** (optional, only if your model needs preprocessing)  

### Model Performance
✅ **Metrics display** (Accuracy, Precision, Recall, F1, AUC) using `st.metric` cards  
✅ **Feature importance** chart from `feature_importance.csv`  

---

## 🧭 App Navigation (Inside UI)

### Sidebar
- **Data Source Selector**
  - SIMULATION (live patients)
  - REAL DATASET (CSV + ML)

### Pages (option_menu)
1. **Dashboard** — Cohort overview + patient list + NEWS2 distribution
2. **Patient Monitor** — Gauges (HR, BP, SpO₂, RR, Temp) + trend charts
3. **Invigilator** — Live edit patient vitals (simulation mode only)
4. **Model Performance** — Metrics cards + feature importance (real dataset mode only)
5. **Model Manager** — Load/upload ML models + scaler + features

---

## 🎨 Key Features

| Feature | Simulation | Real Dataset |
|---------|-----------|--------------|
| Live patients | ✅ Yes (synthetic) | ✅ Yes (CSV) |
| Gauges & charts | ✅ Yes | ✅ Yes |
| ML predictions | ❌ No | ✅ Yes (if model loaded) |
| Edit vitals | ✅ Yes (Invigilator) | ❌ No |
| Timeline control | ✅ Yes (speed/position) | ❌ No |
| Model manager | ✅ Available | ✅ Available |

---

## ⚡ Simulation Features Explained

### Timeline Controls (Sidebar)
- **Simulation Length** (60–1440 min) — Total runtime
- **Manual Position** (slider) — Jump to any minute
- **Simulation Speed** (1–15 min/refresh) — How fast time advances
- **Refresh Rate** (300–3000 ms) — UI update frequency
- **Noise Level** (0.0–0.5) — Vital sign randomness

### Simulation Controls (Sidebar Buttons)
- **▶ Start** — Begin auto-running simulation
- **⏸ Pause** — Pause simulation (keep current state)
- **⟳ Reset** — Clear history, reset all patients, start from minute 0

---

## 📊 Clinical Scores (Automatic)

The app calculates:
- **NEWS2** (National Early Warning Score 2) — 0–20 scale, predicts deterioration
- **qSOFA** (Quick Sequential Organ Failure Assessment) — 0–3 scale, sepsis risk
- **Shock Index** (HR/SBP) — Cardiovascular stress indicator

Risk tiers:
- 🟢 **Low** (NEWS2 ≤ 4)
- 🟡 **Medium** (NEWS2 5–6)
- 🔴 **High** (NEWS2 ≥ 7)

---

## 🔧 Invigilator Mode (Simulation Only)

Edit live:
- Patient profile (stable, sepsis, cardiac, respiratory)
- Age
- Vitals (temp, HR, RR, SpO₂, SBP)
- Flags (supplemental O₂, altered mentation)

Click **Apply changes** to save edits back to simulation.

---

## 💾 Data Files Needed

### For SIMULATION mode
✅ **None required** — App generates synthetic patients

### For REAL DATASET mode
✅ **Required**: `engineered_features.csv`  
- Columns: `timestamp`, `patient_id`, `temp`, `hr`, `rr`, `spo2`, `sbp`, `news2_score`, `is_high_risk`, etc.

✅ **Optional but recommended**:
- `feature_names.json` — List of feature names for ML inference
- `model_metrics.json` — Model performance (accuracy, precision, recall, F1, AUC)
- `feature_importance.csv` — Feature importance for visualization

### For Model Manager
✅ **Optional**:
- `gb_model.pkl`, `rf_model.pkl`, `nn_model.pkl` — Pre-trained models
- `gb_scaler.pkl` — Fitted scaler (only if your model uses StandardScaler)

---

## 🎓 Learning Path

1. **Run simulation first** (requires no files)
   - Play with timeline controls
   - Use Invigilator to edit vitals
   - Understand clinical scores

2. **Then try real dataset** (requires `engineered_features.csv`)
   - See how real data looks in the dashboard
   - Monitor individual patients

3. **Load ML models** (use Model Manager)
   - Upload your trained model + features
   - See predictions in Patient Monitor
   - View performance metrics

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App starts without errors (`streamlit run streamlit_app.py`)
- [ ] Simulation mode runs (press ▶ Start, see charts update)
- [ ] Invigilator can edit data
- [ ] (Optional) CSV file loads in real dataset mode
- [ ] (Optional) Model loads in Model Manager

---

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit_option_menu` | `pip install streamlit-option-menu==0.3.6` |
| Dataset mode shows errors | Put CSV file in same folder as `streamlit_app.py` |
| Metrics show 0 | Load `model_metrics.json` in Model Manager or create it |
| App runs slowly | Reduce refresh rate (↑ ms) or disable real-time updates |

---

## 📖 Next Steps

1. Read **QUICKSTART.md** (5 min setup)
2. Run the app: `streamlit run streamlit_app.py`
3. Read **README.md** for architecture + customization
4. Read **REQUIREMENTS.md** for dependency details

---

## 📞 Support

- **Streamlit docs**: https://docs.streamlit.io
- **Plotly charts**: https://plotly.com/python
- **scikit-learn**: https://scikit-learn.org

---

**Version**: 2.0 (Simulation + Invigilator + Model Manager + Metrics)  
**Updated**: January 2026  
**Status**: Production-Ready ✅
