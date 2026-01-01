# QUICKSTART — Virtual ICU AI Monitor 🏥

Get the app running in **5–10 minutes**.

---

## ⚡ Express Setup (Copy-Paste)

### Windows

```bash
# 1. Create virtual environment
python -m venv vicu_env
vicu_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

### Mac/Linux

```bash
# 1. Create virtual environment
python3 -m venv vicu_env
source vicu_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

Your browser should open automatically to `http://localhost:8501`.

---

## ✅ First-Run Checklist

After the app loads, verify these work:

- [ ] **Sidebar exists** with "Data Source" dropdown
- [ ] **Simulation mode** selected (default)
- [ ] **Navigation menu** shows: Dashboard, Patient Monitor, Invigilator, Model Performance, Model Manager
- [ ] **Dashboard page** displays patient list
- [ ] **Press ▶ Start button** → charts begin updating
- [ ] **Charts update** every ~1 second (refresh rate adjustable)
- [ ] **Patient Monitor page** shows gauges + trend charts
- [ ] **Invigilator page** shows editable patient table
- [ ] **Model Manager page** loads without errors

---

## 🎮 Quick Play Session (2 min)

1. **Start simulation**
   - Press ▶ in sidebar
   - Watch charts update in real-time

2. **Navigate patient**
   - Go to Patient Monitor page
   - See gauges for HR, BP, SpO₂, Respiratory Rate, Temperature
   - Watch clinical scores (NEWS2, qSOFA, Shock Index)

3. **Edit a patient**
   - Go to Invigilator page
   - Click a cell in the table
   - Change a value (e.g., HR from 78 to 120)
   - Click "Apply changes"
   - Go back to Patient Monitor → see updated vital reflected instantly

4. **Explore controls**
   - Pause/resume simulation (⏸ button)
   - Adjust speed (Simulation Speed slider)
   - Jump to different time (Manual Position slider)
   - Reset simulation (⟳ button)

---

## 📖 Understanding the Simulation

Each patient has a **profile**:
- **Stable** — Vitals stay normal
- **Developing Sepsis** — Temperature ↑, HR ↑, BP ↓
- **Cardiac Risk** — HR ↑↑, BP ↓↓
- **Respiratory Decline** — RR ↑, SpO₂ ↓

You can see these profiles in the patient list and edit them with Invigilator.

---

## 🔴 🟡 🟢 Understanding Risk Tiers

The app automatically calculates **NEWS2 score** (0–20):

| Score | Tier | Meaning |
|-------|------|---------|
| 0–4 | 🟢 Low | Patient is stable |
| 5–6 | 🟡 Medium | Monitor closely |
| 7+ | 🔴 High | Requires urgent attention |

---

## ❌ Common Errors & Quick Fixes

### Error: `ModuleNotFoundError: No module named 'streamlit_option_menu'`

**Cause**: Menu component not installed

**Fix**:
```bash
pip install streamlit-option-menu==0.3.6
```

Then run again:
```bash
streamlit run streamlit_app.py
```

---

### Error: `ModuleNotFoundError: No module named 'pandas'` (or numpy, sklearn, etc.)

**Cause**: Dependencies not installed

**Fix**:
```bash
pip install -r requirements.txt
```

---

### Error: `FileNotFoundError: engineered_features.csv` (in Real Dataset mode)

**Cause**: You switched to REAL DATASET mode but the CSV file is missing

**Fix**:
- Stay in SIMULATION mode (default), OR
- Create a CSV file named `engineered_features.csv` in the same folder as `streamlit_app.py`

---

### App Loads But Charts Don't Update

**Cause**: Refresh rate too slow or simulation not started

**Fix**:
- Press ▶ **Start** button in sidebar
- Lower "Refresh Rate (ms)" to 500–1000ms
- Wait a few seconds for charts to populate

---

### App Opens But Sidebar is Empty

**Cause**: Streamlit session not initialized

**Fix**:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

If still broken, restart:
```bash
# Stop the app (Ctrl+C)
# Then restart
streamlit run streamlit_app.py
```

---

### Installation Hangs or Very Slow

**Cause**: TensorFlow or other large packages installing

**Fix**:
Option A) Be patient (can take 5+ minutes)

Option B) Skip TensorFlow (if not needed):
```bash
# Edit requirements.txt and comment out:
# tensorflow==2.14.0

# Then install:
pip install -r requirements.txt
```

---

## 🔧 If You Have Data Files

### CSV Mode (Real Dataset)

If you have `engineered_features.csv`:

1. Put it in the same folder as `streamlit_app.py`
2. Select "REAL DATASET (CSV + ML)" from sidebar
3. The Dashboard will load your CSV data
4. Patient Monitor will show your real patients

Required columns in CSV:
- `timestamp`
- `patient_id`
- `temp`, `hr`, `rr`, `spo2`, `sbp`
- `news2_score` (or it will calculate)
- `is_high_risk` (0 or 1)

---

### ML Models (Model Manager)

If you have trained models:

1. Put `.pkl` files in the same folder as `streamlit_app.py`
   - `gb_model.pkl`
   - `feature_names.json`
   - (optional) `gb_scaler.pkl`
   - (optional) `model_metrics.json`

2. Go to **Model Manager** page
3. Click "Load selected local model"
4. Go to **Patient Monitor** → will now show ML predictions

---

## 🧪 Quick Test Without Files

You **don't need any files** to test the app:

```bash
streamlit run streamlit_app.py
```

- Simulation mode works immediately
- Creates 10 synthetic patients
- All charts and controls work
- No CSV/model files needed

---

## 📊 What You're Looking At

### Dashboard Page
- **Metrics cards**: Total patients, High/Medium/Low risk counts
- **Patient list**: All patients with their current vitals
- **NEWS2 distribution**: Histogram showing risk spread

### Patient Monitor Page
- **5 gauges**: HR, BP, SpO₂, RR, Temperature (color-coded zones)
- **Clinical scores**: NEWS2, qSOFA, Shock Index
- **NEWS2 risk meter**: Gauge showing escalating risk
- **Trend charts**: 5 line charts showing vital sign history

### Invigilator Page
- **Editable table**: Click any cell to edit
- **Apply changes**: Saves edits back to simulation
- Works **only in SIMULATION mode**

### Model Manager Page
- **Local models**: Select from gb_model.pkl, rf_model.pkl, nn_model.pkl
- **Upload models**: Drag-drop your `.pkl` files
- **Load button**: Applies the model to the session

### Model Performance Page
- **Metrics cards**: Accuracy, Precision, Recall, F1, AUC (from model_metrics.json)
- **Feature importance**: Bar chart (from feature_importance.csv)
- Works **only in REAL DATASET mode**

---

## 🎯 Next: Deeper Learning

1. ✅ You've done: Quick install & play
2. 📖 **Next**: Read full **README.md** (architecture + features)
3. 🔍 **Then**: Read **REQUIREMENTS.md** (all dependencies)
4. 🛠️ **Finally**: Customize/extend as needed

---

## 💡 Pro Tips

- **Slow simulation?** Increase "Refresh Rate (ms)" to 2000–3000
- **Too fast?** Lower it to 300–500
- **Want more patients?** Edit sidebar code (n_patients parameter)
- **Want different profiles?** Modify make_patient() function in `streamlit_app.py`
- **Want real data?** Load your CSV in Real Dataset mode

---

## 🆘 Still Stuck?

1. Check **REQUIREMENTS.md** (dependency issues)
2. Read **README.md** (full documentation)
3. Run with debug:
   ```bash
   streamlit run streamlit_app.py --logger.level=debug
   ```
4. Check Streamlit docs: https://docs.streamlit.io

---

**You're all set!** 🎉 The app is running. Now explore and have fun!
