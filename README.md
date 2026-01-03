# Virtual ICU (vICU) AI Monitor — Complete Technical Documentation

A **production-ready Streamlit dashboard** for ICU patient monitoring with real-time simulation, ML predictions, and clinical scoring.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Simulation Mode](#simulation-mode)
7. [Real Dataset Mode](#real-dataset-mode)
8. [Model Manager](#model-manager)
9. [Clinical Scoring](#clinical-scoring)
10. [Customization](#customization)
11. [Troubleshooting](#troubleshooting)

---

## Overview

**Virtual ICU** is a multi-patient monitoring dashboard that supports:

1. **Live Simulation** — Generates 10+ synthetic ICU patients with realistic vital sign drift
2. **Real Data** — Loads CSV data with engineered features
3. **ML Predictions** — Loads trained models for disease risk prediction
4. **Clinical Scores** — Automatically calculates NEWS2, qSOFA, Shock Index
5. **Interactive Charts** — Real-time Plotly visualizations
6. **Data Editing** — Invigilator panel to modify vitals live

**Use Cases:**
- Medical education (student training)
- Model evaluation (test ML predictions)
- Clinical decision support (patient risk assessment)
- Simulation-based research (test hypotheses)

---

## Key Features

### Simulation Mode ✅

- **Multi-patient generation** (10 profiles, configurable)
- **4 patient profiles**: Stable, Developing Sepsis, Cardiac Risk, Respiratory Decline
- **Real-time vital sign drift** with Gaussian noise
- **Live gauges** (5 clinical parameters)
- **Clinical scores** calculated automatically
- **Invigilator panel** to edit vitals on-the-fly
- **Timeline controls**: seek, speed, refresh rate
- **Full chart history** for each patient

### Real Dataset Mode ✅

- **Load CSV data** (`engineered_features.csv`)
- **Patient selection** dropdown
- **Patient trend visualization** with Plotly
- **Optional ML predictions** (if model is loaded)

### Model Manager ✅

- **Load local models** (gb/rf/nn)
- **Upload custom models** (.pkl files)
- **Optional scaler support** (StandardScaler/MinMaxScaler)
- **Feature list management** (feature_names.json)
- **Session persistence** (models stay loaded until changed)

### Model Performance ✅

- **Metrics display** (st.metric cards):
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC
- **Feature importance chart** (Plotly bar chart)
- **Multi-model comparison** (switch between models)

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ SIMULATION MODE                  REAL DATASET MODE          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  make_patient() ──→ sim_df() ────→ Dashboard               │
│  step_sim()      │   (clinical     Patient Monitor          │
│                  │    scores)      Invigilator              │
│                  │                                          │
│                  └──→ st.session_state                      │
│                       (history)                              │
│                                                              │
│                      CSV ──→ engineered_features.csv ────→  │
│                             Patient Monitor                 │
│                             Model Performance               │
│                             (if model loaded)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

| Module | Purpose |
|--------|---------|
| `news2_score()` | Calculate NEWS2 early warning score |
| `qsofa_score()` | Calculate qSOFA sepsis score |
| `shock_index()` | Calculate cardiovascular stress (HR/SBP) |
| `gauge()` | Render Plotly gauge chart |
| `news2_risk_meter()` | Risk indicator gauge |
| `make_patient()` | Generate synthetic patient |
| `init_sim()` | Initialize simulation |
| `step_sim()` | Advance simulation by 1 tick |
| `sim_df()` | Generate current DataFrame |
| `apply_invigilator_edits()` | Apply UI edits to simulation |
| `load_local_assets()` | Load .pkl and .json files (cached) |
| `ml_predict()` | Make predictions with loaded model |

### State Management

Uses `st.session_state` for:
- `sim_patients` — List of patient objects (simulation)
- `sim_history` — Dict of patient vital sign histories
- `sim_position` — Current simulation time (minutes)
- `ml_model`, `ml_scaler`, `ml_feature_names` — Loaded ML assets
- `selected_patient` — Currently viewed patient

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# 1. Clone/download project
cd virtual-icu

# 2. Create virtual environment (recommended)
python -m venv vicu_env

# Activate:
# Windows: vicu_env\Scripts\activate
# Mac/Linux: source vicu_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import streamlit; print('✓ OK')"

# 5. Run
streamlit run streamlit_app.py
```

---

## Usage Guide

### Starting the App

```bash
streamlit run streamlit_app.py
```

Browser opens to `http://localhost:8501`

### Sidebar Controls

**Data Source**
- SIMULATION (live patients)
- REAL DATASET (CSV + ML)

**Analysis Window**
- Slider to adjust time window for charts (5–60 min)

**Simulation Timeline** (only in SIMULATION mode)
- Simulation Length (60–1440 min)
- Simulation Speed (1–15 min/refresh)
- Refresh Rate (300–3000 ms)
- Noise Level (0.0–0.5)
- Manual Position (seek slider)

**Simulation Controls**
- ▶ Start — Begin simulation
- ⏸ Pause — Pause (keep state)
- ⟳ Reset — Clear all, restart from minute 0

---

## Simulation Mode

### How It Works

1. **Initialization**
   - 10 patients created with random profiles
   - Each patient has vitals + drift parameters

2. **Patient Profiles**

| Profile | Temp | HR | RR | SpO₂ | SBP | Drift |
|---------|------|----|----|------|-----|-------|
| Stable | 36.9 | 78 | 14 | 97 | 128 | None (0) |
| Sepsis | 38.3 | 105 | 22 | 94 | 105 | Temp↑, HR↑, BP↓ |
| Cardiac | 37.2 | 115 | 18 | 95 | 100 | HR↑↑, BP↓↓ |
| Respiratory | 37.1 | 92 | 24 | 92 | 120 | RR↑, SpO₂↓ |

3. **Each Tick**
   - Vitals drift by parameters + Gaussian noise
   - Clinical scores recalculated
   - History appended to session_state
   - UI reruns with `st.rerun()`

4. **Controls**
   - **Timeline**: Seek to any minute (recalculates from start)
   - **Speed**: Control simulation time acceleration (1–15 min/tick)
   - **Refresh**: Control UI update frequency (300–3000 ms)

### Pages

#### Dashboard
- Metrics: Patient count, high/medium/low risk
- Patient table: All patients with current vitals
- NEWS2 distribution histogram

#### Patient Monitor
- 5 gauges (color-coded)
- Clinical scores (NEWS2, qSOFA, Shock Index)
- NEWS2 risk meter
- Trend charts (5 line plots for vital history)

#### Invigilator
- Editable table (st.data_editor)
- Change: profile, age, vitals, flags
- Click "Apply changes" to save back

---

## Real Dataset Mode

### CSV Requirements

File: `engineered_features.csv`

Required columns:
```csv
timestamp,patient_id,temp,hr,rr,spo2,sbp,news2_score,is_high_risk,qsofa_score,shock_index
```

Optional columns:
- age
- profile (for tracking patient type)
- supp_o2 (supplemental oxygen flag)
- altered_mentation (cognitive status flag)

### Loading Data

1. Place `engineered_features.csv` in project root
2. Select "REAL DATASET (CSV + ML)" from sidebar
3. Dashboard loads data automatically

### Patient Selection

Dashboard shows:
- Total patient count
- High-risk patient count
- Average NEWS2 score

Patient Monitor:
- Dropdown to select patient
- Gauges + trend charts
- ML prediction (if model loaded)

---

## Model Manager

### Loading Local Models

1. Go to **Model Manager** page
2. Select model from dropdown: gb_model.pkl, rf_model.pkl, nn_model.pkl
3. Specify paths (or leave default):
   - Scaler file (optional): `gb_scaler.pkl`
   - Feature names: `feature_names.json`
   - Metrics (optional): `model_metrics.json`
4. Click "Load selected local model"

### Uploading Custom Models

1. Prepare files:
   - `your_model.pkl` (scikit-learn compatible)
   - `feature_names.json` (list of features, JSON array)
   - (optional) `your_scaler.pkl`
   - (optional) `metrics.json`

2. Drag-drop in upload boxes
3. Click "Load uploaded model"

### Feature Names Format

File: `feature_names.json`
```json
[
  "temp",
  "hr",
  "rr",
  "spo2",
  "sbp",
  "age",
  "news2_score",
  "qsofa_score",
  "shock_index"
]
```

### Metrics Format

File: `model_metrics.json`
```json
{
  "gradient_boosting": {
    "accuracy": 0.92,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.865,
    "auc": 0.945
  },
  "random_forest": {
    "accuracy": 0.90,
    ...
  }
}
```

---

## Clinical Scoring

### NEWS2 (National Early Warning Score 2)

**Range**: 0–20 (higher = worse)

Scoring:
- Temperature: -3 to +2 points
- Respiratory Rate: -3 to +3 points
- Oxygen Saturation: 0 to +3 points
- Systolic BP: 0 to +3 points
- Heart Rate: -3 to +3 points
- Consciousness: 0 or +3 points
- Supplemental O₂: 0 or +2 points

**Risk Tiers:**
- 0–4: Low (🟢)
- 5–6: Medium (🟡)
- 7+: High (🔴)

### qSOFA (Quick Sequential Organ Failure Assessment)

**Range**: 0–3

Criteria:
- RR ≥ 22: +1 point
- SBP ≤ 100: +1 point
- Altered mentation: +1 point

**Clinical Use:** Sepsis screening

### Shock Index

**Formula**: Heart Rate / Systolic BP

**Interpretation:**
- < 0.5: Normal
- 0.5–0.9: Compensated
- > 1.0: Decompensated (shock risk)

---

## Customization

### Add More Patients

In sidebar initialization:
```python
if st.button("⟳", ...):
    st.session_state.sim_patients = init_sim(n_patients=20)  # Change 10 to 20
```

### Add New Patient Profile

In `make_patient()`:
```python
elif profile == "my_profile":
    vitals = dict(temp=37.0, hr=80, ...)
    drift = dict(temp=0.005, hr=0.02, ...)
```

### Adjust Simulation Speed

In sidebar:
```python
st.session_state.sim_step_minutes = st.slider(
    "Simulation Speed",
    1, 30,  # Change range
    int(st.session_state.sim_step_minutes)
)
```

### Add More Clinical Scores

Create new scoring function:
```python
def my_score(temp, hr, ...):
    score = 0
    # your logic
    return int(score)
```

Then call in `sim_df()`:
```python
my_result = my_score(v["temp"], v["hr"], ...)
```

### Modify Gauges

Edit `gauge()` calls with different `steps`:
```python
gauge("Heart Rate", current["hr"], "bpm", 40, 160,
      [{"range":[40,60],"color":"#4facfe"},
       {"range":[60,100],"color":"#51CF66"},
       {"range":[100,130],"color":"#FFA500"},
       {"range":[130,160],"color":"#FF6B6B"}])
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: streamlit_option_menu`
```bash
pip install streamlit-option-menu==0.3.6
```

**Issue**: Charts don't update in simulation
- Press ▶ **Start** button
- Lower "Refresh Rate (ms)"
- Check browser console (F12)

**Issue**: Real Dataset mode shows no data
- Check filename: must be exactly `engineered_features.csv`
- Check location: same folder as `streamlit_app.py`
- Check columns: must include `patient_id`, `timestamp`, `temp`, `hr`, `rr`, `spo2`, `sbp`, `news2_score`, `is_high_risk`

**Issue**: Model Manager won't load model
- Verify `feature_names.json` exists and is valid JSON
- Check that model is scikit-learn compatible (.pkl format)
- Scaler is optional—leave empty if not needed

**Issue**: Metrics show 0 in Model Performance
- Check `model_metrics.json` exists
- Verify structure matches expected format
- Or load metrics via Model Manager upload

---

## Performance Notes

### Optimization

- Cached resources: `@st.cache_resource` for ML models
- Limited history: Patient histories capped at 400 records
- Efficient DataFrame operations: Pandas for CSV handling

### Scaling

For 100+ patients:
- Increase refresh rate (slower updates)
- Reduce history size
- Use downsampling for charts

---

## Security Notes

- **Pickle files**: Only load models from trusted sources (pickle can execute code)
- **CSV files**: Validate data before using in clinical decisions
- **Session state**: Local only, not persistent across server restarts

---

## Future Enhancements

- [ ] Database persistence (PostgreSQL/SQLite)
- [ ] Multi-user sync (WebSocket)
- [ ] User authentication
- [ ] Alert system (thresholds)
- [ ] Export to PDF/CSV
- [ ] Waveform data (ECG, SpO₂ waveforms)
- [ ] Predictive alerts (ML-based)
- [ ] Integration with EHR systems

---

## References

- Streamlit: https://docs.streamlit.io
- Plotly: https://plotly.com/python
- scikit-learn: https://scikit-learn.org
- NEWS2 Scoring: https://www.rcplondon.ac.uk
- qSOFA: https://www.sepsis.org

---

## License

Add your license (MIT/Apache-2.0/etc.)

---

**Version**: 2.0 (Full-featured production release)  
**Updated**: January 2026  
**Status**: ✅ Production-Ready
