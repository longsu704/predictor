import math
import os
import sys
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
     sys.path.insert(0, _HERE)

from predictor1_prep import RadiopharmPreprocessor


# Part1. Basic configuration file （第一部分：基础配置文件）

st.set_page_config(
    page_title="Radiopharmaceutical Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💊",
)

DEPLOY_DIR = r'D:\wordjihe\deploy'
FALLBACK_DIRS = [
    DEPLOY_DIR,
    r'D:\wordjihe',
    _HERE,
]

# Distribution prediction module. It is divided into two model dimensions: CFM and EFM, as well as three partitioning scheme dimensions （CFM+EFM以及三种划分）

MODEL_KEYS = {
    'CFM': 'Core Feature Model',
    'EFM': 'Extended Feature Model',
}
MODEL_LABELS = {
    'CFM': 'Core Feature Model',
    'EFM': 'Extended Feature Model',
}
SCHEME_LABELS = {
    'Random':   'Random Split',
    'Compound': 'Leave Compound Out',
    'Group':    'Leave Publication Out',
}
MODEL_ORDER = ['CFM', 'EFM']
SCHEME_ORDER = ['Random', 'Group', 'Compound']
DEFAULT_MODEL = 'CFM'
DEFAULT_SCHEME = 'Random'

# The judgment criteria for determining PSMA-617 are time and tumor uptake
THRESHOLD_MAP = {'1': 14.5, '4': 18.1, '24': 13.8}

MODEL_FILE_TMPL = 'predictor1_{model}_{scheme}_model.pkl'


# Part2: Basic style of web page (第二部分：网页端样式)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Arial', Times, serif;
        font-size: 15px;
        color: #212529;
        background-color: #f8f9fa;
    }
    h1 {
        color: #1a476f; font-weight: bold; font-size: 26px;
         border-bottom: 3px solid #2c5282; padding-bottom: 10px; margin-bottom: 20px;
    }
    h2 { color: #2c5282; font-weight: 600; font-size: 22px; margin-top: 25px; margin-bottom: 15px; }
    h3 { color: #4a5568; font-weight: 600; font-size: 18px; margin-bottom: 10px; }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border: 1px solid #ced4da; border-radius: 4px; padding: 8px 12px;
    }
    .stButton > button {
         background-color: #2c5282; color: white; border: none; border-radius: 4px;
        padding: 10px 20px; font-weight: 600; transition: all 0.2s ease;
    }
    .stButton > button:hover { background-color: #1a365d; transform: translateY(-1px); }
    .result-card {
        background-color: white; border-radius: 8px; padding: 20px;
         box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #2c5282; margin: 15px 0;
    }
    .eval-card {
        background-color: white; border-radius: 8px; padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #28a745; margin: 15px 0;
    }
    .eval-card.warning { border-left: 4px solid #dc3545; }
    .dataframe { border: 1px solid #dee2e6; border-radius: 4px; }
     .divider { border-top: 1px solid #dee2e6; margin: 20px 0; }
    .history-item { padding: 10px; border-bottom: 1px solid #f1f3f5; }
    .history-item:last-child { border-bottom: none; }
     .scheme-card {
         padding: 14px 18px; border-radius: 8px; border: 2px solid #dee2e6;
        background-color: white; border-left: 4px solid #2c5282; margin: 10px 0;
    }
    [data-testid="stSidebar"] { background-color: #f1f3f5; padding-top: 2rem; }
     [data-testid="stSidebarNav"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)



# session state （缓存会话）

if 'selected_model_key' not in st.session_state:
     st.session_state.selected_model_key = DEFAULT_MODEL
if 'selected_scheme' not in st.session_state:
    st.session_state.selected_scheme = DEFAULT_SCHEME
if 'prediction_history' not in st.session_state:
      st.session_state.prediction_history = []
if 'calculation_history' not in st.session_state:
     st.session_state.calculation_history = []
if 'hed_calculation_history' not in st.session_state:
     st.session_state.hed_calculation_history = []


# Sidebar navigation settings （侧边栏导航）

st.sidebar.title("📋 Function Navigation")
selected_module = st.sidebar.selectbox(
    "Select Function Module",
    options=[
        "Radiopharmaceutical Biodistribution Prediction",
        "Radioactive Drug Injection Dose Calculator",
    ],
    index=0,
)


# Part3: Distribution prediction module (生物分布预测模块)

def _resolve_model_path(filename):
    for d in FALLBACK_DIRS:
        if not d:
            continue
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_key, scheme):

    filename = MODEL_FILE_TMPL.format(model=model_key.lower(), scheme=scheme.lower())
    path = _resolve_model_path(filename)
    if path is None:
        return None, (f"Model file not found: {filename} (searched: "
                      + ", ".join([d for d in FALLBACK_DIRS if d]) + ")")
    try:
        obj = joblib.load(path)
    except Exception as exc:                       # noqa: BLE001
        return None, f"Failed to read {filename}: {exc}"

    if isinstance(obj, dict):
        meta = dict(obj)
    else:
        meta = {'cat_cols': [], 'num_cols': [], 'cat_levels': {}, 'cv_metrics': None,
                 'log_mode': 'ln1p', 'trained_rows': None, 'cv_desc': 'n/a',
                'pipeline': obj, 'model_key': model_key, 'scheme': scheme,
                 'model_family': type(obj).__name__, 'is_reference': False}
    meta['_path'] = path
    if 'pipeline' not in meta:
        return None, f"{filename} does not contain a 'pipeline' object."
    return meta, None


def get_pipeline(meta):
    obj = meta['pipeline']
    return obj if hasattr(obj, 'named_steps') else None


def _inverse(z, mode):
    # log scale （转换为log尺度）
    z = float(np.clip(z, -50.0, 50.0))
    if mode == 'ln1p':
        return float(max(np.expm1(z), 0.0))
    if mode == 'log10':
        return float(10.0 ** z)
    if mode == 'log10p':
        return float(max(10.0 ** z - 1.0, 0.0))
    return float(max(np.expm1(z), 0.0))


def predict_one(meta, input_dict):
    #  Return to the original scale for log scale( 返回为原始尺度)
    pipe = get_pipeline(meta)
    if pipe is None:
        raise RuntimeError(
             "The loaded model object is a bare estimator without preprocessing. "
            "Please provide the full sklearn Pipeline (as saved by export_predictor1.py).")
    cat_cols, num_cols = meta['cat_cols'], meta['num_cols']
    row = {c: input_dict.get(c, 'NA') for c in cat_cols}
    for c in num_cols:
        v = input_dict.get(c, np.nan)
        row[c] = np.nan if v is None else v
    X = pd.DataFrame([row])[list(cat_cols) + list(num_cols)]
    pred_log = float(np.ravel(pipe.predict(X))[0])
    return _inverse(pred_log, meta.get('log_mode', 'ln1p')), pred_log

 # Dose optimization module （剂量寻优，步长为0.5，从0.5到10）
def optimize_dosage(meta, input_dict, threshold):

    dosage_col = 'Injection Dosage'
    if dosage_col not in meta['num_cols']:
        st.warning("Injection dose field not found, dose optimization cannot be performed.")
        return None, []

    attempts, recommended = [], None
    for dosage in np.arange(0.5, 10.1, 0.5):
        new_input = dict(input_dict)
        new_input[dosage_col] = float(dosage)
        try:
            pred_orig, _ = predict_one(meta, new_input)
        except Exception:
            continue
        meets = bool(pred_orig > threshold)
        attempts.append({'Dosage': round(float(dosage), 1),
                         'Prediction Value (% ID/g)': round(pred_orig, 4),
                         'Meets Standard': meets})
        if meets and recommended is None:
            recommended = float(dosage)
    return recommended, attempts


#Part4 ：Biological distribution prediction module （核心：生物分布的界面）
if selected_module == "Radiopharmaceutical Biodistribution Prediction":
    st.title("Radiopharmaceutical Biodistribution Prediction Platform")
    st.markdown("*A machine learning web page based on radiopharmaceutical properties "
                "to predict tumor biodistribution in mice*")

    #The model selection area can be divided into two drop-down boxes, allowing for the selection of both the model and the grouping method（下拉选择框）
    st.subheader("1. Model Selection")

    col_m, col_s = st.columns(2, gap="large")
    with col_m:
        model_key = st.selectbox(
            "Select model (feature set)",
            options=MODEL_ORDER,
            index=MODEL_ORDER.index(st.session_state.selected_model_key),
            format_func=lambda k: MODEL_LABELS[k],
            key='model_selectbox',
            help="CFM = Core Feature Model; "
                 "EFM = Extended Feature Model.",
        )
    with col_s:
        scheme = st.selectbox(
            "Select validation scheme",
            options=SCHEME_ORDER,
            index=SCHEME_ORDER.index(st.session_state.selected_scheme),
            format_func=lambda k: SCHEME_LABELS[k],
            key='scheme_selectbox',
            help="All options are XGBoost models; they differ only in how the data was "
                 "split when the model was trained and validated.",
        )
    st.session_state.selected_model_key = model_key
    st.session_state.selected_scheme = scheme

    meta, load_error = load_model(model_key, scheme)

    if meta is not None:
        _desc = {
            'Random': "Trained and validated with a <b>random split</b> — the sample space "
                      "is interpolative (rows from the same publication may appear in both "
                      "training and test sets). Highest expected accuracy.",
            'Compound': "Trained and validated with a <b>compound-grouped split</b> — no "
                        "compound appears in both training and test sets. Tests "
                        "extrapolation to new compounds.",
            'Group': "Trained and validated with a <b>publication-grouped split</b> — no "
                     "reference appears in both training and test sets. Strictest setting; "
                     "tests extrapolation to new publications.",
        }[scheme]
        _extra = "XGBoost regression"
        if meta.get('is_reference'):
            _extra += " · reference model exported by export_predictor1.py"
        else:
            _extra += f" · {meta.get('model_family', 'estimator')}"
        st.markdown(f"""
        <div class="scheme-card">
            <h3 style='margin:0 0 6px 0;'>{MODEL_LABELS[model_key]} — {SCHEME_LABELS[scheme]}</h3>
            <p style='color:#4a5568; margin:0 0 6px 0;'>{_desc}</p>

        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Failed to load model components. {load_error}")
        st.markdown("""
**How to fix:** run the export script once to create the model files —

```bash
python export_predictor1.py
```

It writes six files (`predictor1_cfm_*_model.pkl` and `predictor1_efm_*_model.pkl`)
into `D:\\wordjihe\\deploy`.
If you already have your own exported models, simply place them there with the
same file names; `predictor1_prep.py` must stay next to this script, because the
saved pipelines reference it when they are loaded.
""")
        st.stop()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Input parameters

    cat_cols = meta['cat_cols']
    num_cols_input = meta['num_cols']
    cat_levels = meta.get('cat_levels', {})

    # A globally persistent dictionary that can operate normally even when switching between CFM and EFM
    store = st.session_state.setdefault('persist_inputs', {})

    st.subheader("2. Input Parameters")
    input_dict = {}
    col_cat, col_num = st.columns(2, gap="large")

    with col_cat:
        st.markdown("<h3 style='font-size:16px;'>Categorical Features</h3>",
                    unsafe_allow_html=True)
        for col in cat_cols:
            options = cat_levels.get(col, [])
            wkey = f"in_{col}"
            if not options:
                st.warning(f"No candidate values recorded for {col}; please type it manually.")
                st.session_state.setdefault(wkey, store.get(col, 'NA'))
                val = st.text_input(label=f"{col}", key=wkey)
                store[col] = val
                input_dict[col] = val
                continue

            if wkey not in st.session_state:
                prev = store.get(col)
                st.session_state[wkey] = prev if prev in options else options[0]
            val = st.selectbox(
                label=f"{col}", options=options, key=wkey,
                help=f"Select value for {col} (pre-sorted alphabetically)")
            store[col] = val
            input_dict[col] = val

    with col_num:
        st.markdown("<h3 style='font-size:16px;'>Numerical Features</h3>",
                    unsafe_allow_html=True)
        for col in num_cols_input:
            rng = (meta.get('num_ranges') or {}).get(col)
            help_txt = "Enter numerical value or 'NA' for missing data"
            if rng:
                help_txt += (f" | observed range in training data: {rng[0]:g} "
                             f"(median {rng[1]:g}) ~ {rng[2]:g}")
            wkey = f"in_{col}"
            if wkey not in st.session_state:
                st.session_state[wkey] = store.get(col, "NA")
            val_str = st.text_input(label=f"{col}", key=wkey, help=help_txt)
            store[col] = val_str
            if str(val_str).strip().upper() == "NA":
                input_dict[col] = np.nan
            else:
                try:
                    input_dict[col] = float(val_str)
                except ValueError:
                    st.warning(f"Invalid input for {col} - using NA instead")
                    input_dict[col] = np.nan

    # prediction result
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("3. Prediction & Results")

    if st.button("Run Prediction", type="primary"):
        with st.spinner('Processing data...'):
            input_df = pd.DataFrame([input_dict])
            st.markdown("<h3 style='font-size:16px;'>User Input Data</h3>",
                        unsafe_allow_html=True)
            input_display = input_df.copy().replace({np.nan: "NA"})
            st.dataframe(
                input_display.style.set_properties(**{
                    'background-color': 'white', 'border': '1px solid #dee2e6',
                    'padding': '8px'}),
                use_container_width=True)

            try:
                prediction, prediction_log = predict_one(meta, input_dict)
            except Exception as exc:               # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                st.stop()

            st.session_state.prediction_history.insert(0, {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': f"{MODEL_LABELS[model_key]} / {SCHEME_LABELS[scheme]}",
                'prediction': round(prediction, 4),
                'input_data': dict(input_dict),
            })
            st.session_state.prediction_history = st.session_state.prediction_history[:10]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="result-card">
                <h3 style='margin:0;'>Prediction Result</h3>
                <p style='font-size:18px; font-weight:bold; color:#2c5282; margin:10px 0;'>
                    {:.4f} % ID/g
                </p>
                <p style='color:#6c757d; margin:0 0 6px 0;'>
                    Model: {} | Split scheme: {} | Calculation Time: {}
                </p>
                <p style='color:#6c757d; margin:0; font-size:12px;'>
                    (log-scale value: {:.4f})
                </p>
            </div>
            """.format(prediction, model_key.upper(), scheme,
                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       prediction_log), unsafe_allow_html=True)

            # Determine whether it exceeds the standard reference（判别参考）
            nuclide_val = target_val = time_val = chelating_agent_val = ""
            for k, v in input_dict.items():
                k_lower = str(k).lower()
                if k_lower == "nuclide":
                    nuclide_val = str(v).strip()
                if "target" in k_lower:
                    target_val = str(v).strip()
                if "time" in k_lower and pd.notna(v):
                    time_val = str(int(float(v))).strip()
                if "chelating agent" in k_lower or k_lower in ("chelatingagent", "chelating"):
                    chelating_agent_val = str(v).strip()

            # CFM does not contain chelating agents
            # EFM contains chelating agents, so it is necessary to set up a separate column for chelating agents
            if model_key == 'CFM':
                condition_met = (nuclide_val == "177Lu" and target_val == "PSMA"
                                 and time_val in THRESHOLD_MAP)
            else:
                condition_met = (nuclide_val == "177Lu" and target_val == "PSMA"
                                 and chelating_agent_val == "DOTA"
                                 and time_val in THRESHOLD_MAP)

            if condition_met:
                threshold = THRESHOLD_MAP[time_val]
                original_dosage = input_dict.get("Injection Dosage", "NA")
                if not pd.notna(original_dosage):
                    original_dosage = "NA"

                rec_dosage, dosage_attempts = optimize_dosage(meta, input_dict, threshold)

                if prediction > threshold:
                    eval_result = "Excellent"
                    eval_remark = (f"Current dose({original_dosage}),{time_val}h Uptake"
                                   f"（{prediction:.4f}）,better than Lu-177-PSMA617"
                                   f"（threshold：{threshold}）。")
                    card_class = "eval-card"
                else:
                    eval_result = "Not up to standard"
                    eval_remark = (f"Current dose({original_dosage}),{time_val}h Uptake"
                                   f"（{prediction:.4f}）,not up to standard"
                                   f"（threshold：{threshold}）.")
                    if rec_dosage:
                        eval_remark += (f" The predicted minimum effective dose is "
                                        f"{rec_dosage} (same unit as the dose input).")
                    card_class = "eval-card warning"

                st.markdown(f"""
                <div class="{card_class}">
                    <h3 style='margin:0;'>PSMA617 Excellence Standard Evaluation</h3>
                    <p style='font-size:16px; font-weight:bold; margin:10px 0;'>
                        Evaluation: {eval_result}
                    </p>
                    <p style='font-size:14px; color:#4a5568; margin:10px 0;'>
                        Remark: {eval_remark}
                    </p>
                    <p style='color:#6c757d; margin:0;'>
                        Evaluation Standard: {time_val}h tumor uptake ≥ {threshold} |
                        Model: {model_key.upper()} ({scheme} split)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if dosage_attempts:
                    st.markdown("<h3 style='font-size:16px;'>Dosage Optimization Attempts "
                                "(Full Range)</h3>", unsafe_allow_html=True)
                    dosage_df = pd.DataFrame(dosage_attempts)
                    dosage_df['Meets Standard'] = dosage_df['Meets Standard'].map(
                        {True: '✅', False: '❌'})
                    st.dataframe(
                        dosage_df.style.set_properties(**{
                            'background-color': 'white', 'border': '1px solid #dee2e6',
                            'padding': '8px'}),
                        use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    "<p style='color:#6c757d;'>The PSMA617 excellence-standard evaluation is "
                    "only available for ¹⁷⁷Lu‑PSMA combinations at 1 / 4 / 24 h "
                    "(plus the DOTA chelator when using the Extended Feature Model), "
                    "so it was skipped for this input.</p>",
                    unsafe_allow_html=True)

    # Historical record（历史记录）
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("4. Recent Prediction History (Last 10)")
    if st.session_state.prediction_history:
        history_data = []
        for entry in st.session_state.prediction_history:
            input_str = ", ".join(
                [f"{k}: {v if not pd.isna(v) else 'NA'}"
                 for k, v in entry['input_data'].items()])
            history_data.append({
                'Time': entry['timestamp'],
                'Model': entry['model'],
                'Prediction Value (% ID/g)': entry['prediction'],
                'Input Summary': input_str[:100] + "..." if len(input_str) > 100 else input_str,
            })
        st.dataframe(
            pd.DataFrame(history_data).style.set_properties(**{
                'background-color': 'white', 'border': '1px solid #dee2e6', 'padding': '8px'}),
            use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.markdown("<p style='color:#6c757d;'>No prediction history yet</p>",
                    unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:50px; padding-top:20px; border-top:1px solid #dee2e6;
                color:#6c757d; text-align:center;'>
        Radiopharmaceutical Biodistribution Prediction Platform | Designed for Academic Standards
    </div>
    """, unsafe_allow_html=True)


# part5：Clinical Translation Calculator（临床转化计算器）

elif selected_module == "Radioactive Drug Injection Dose Calculator":
    FIXED_DOSE_DRUGS = {
        "Custom Fixed": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1.0},
        "¹⁷⁷Lu-PSMA-617": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1000.0},
        "¹⁷⁷Lu‑DOTATATE": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1000.0},
        "¹³¹I‑MIBG": {"physical_half_life": 8.02, "fixed_total_activity": 11100.0, "calibration_concentration": 111.0},
        "⁹⁹ᵐTc‑MDP": {"physical_half_life": 0.2503, "fixed_total_activity": 740.0, "calibration_concentration": 37.0},
        "⁹⁹ᵐTc‑DTPA": {"physical_half_life": 0.2503, "fixed_total_activity": 555.0, "calibration_concentration": 37.0},
        "⁶⁸Ga‑DOTATATE": {"physical_half_life": 0.047, "fixed_total_activity": 150.0, "calibration_concentration": 218.0},
    }
    WEIGHT_BASED_DRUGS = {
        "Custom Parameters": {"physical_half_life": 11.4, "target_dose_per_kg": 0.055, "calibration_concentration": 1.1},
        "²²³RaCl₂": {"physical_half_life": 11.4, "target_dose_per_kg": 0.055, "calibration_concentration": 1.1},
        "²²⁵Ac-PSMA-617": {"physical_half_life": 9.92, "target_dose_per_kg": 0.125, "calibration_concentration": 1.1},
        "²²⁵Ac-DOTATATE": {"physical_half_life": 9.92, "target_dose_per_kg": 0.120, "calibration_concentration": 1.1},
        "²²⁵Ac-PSMA-I&T": {"physical_half_life": 9.92, "target_dose_per_kg": 0.100, "calibration_concentration": 1.1},
        "²²⁵Ac-J591": {"physical_half_life": 9.92, "target_dose_per_kg": 0.035, "calibration_concentration": 1.1},
        "²¹²Pb-DOTATATE": {"physical_half_life": 0.4433, "target_dose_per_kg": 2.5012, "calibration_concentration": 1.1}
    }
    ORGAN_DOSE_TABLE2 = {
        "Lacrimal Gland": 2.1, "Salivary Gland": 0.63, "Kidney": 0.43, "Left Colon": 0.58, "Rectum": 0.56,
        "Right Colon": 0.32, "Bladder Wall": 0.32, "Heart Wall": 0.17, "Liver": 0.09, "Lung": 0.11, "Whole Body": 0.037
    }
    MOUSE_TUMOR_DOSE = {
        "Lacrimal Gland": 24.0, "Salivary Gland": 8.5, "Kidney": 5.8, "Left Colon": 7.2, "Rectum": 6.9,
        "Right Colon": 4.1, "Bladder Wall": 4.1, "Heart Wall": 4.3, "Liver": 2.2, "Lung": 1.8, "Whole Body": 1.4, "Tumor": 30.0
    }

    st.title("💉 Radioactive Drug Calculator")
    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)
    st.subheader("🧪 Extrapolation of Tumor‑Bearing Mouse Dose to Human Equivalent Dose (HED)")
    st.divider()

    hed_method = st.selectbox(
        "Select Extrapolation Method",
        options=[
            "Method 1: Custom BSA Universal Formula",
            "Method 2: Allometric Scaling Law (b=0.67)",
            "Method 3: Tumor Volume Normalization"
        ],
        index=0
    )
    st.caption("📌 Unit: Mouse Dose(MBq/kg) | HED(MBq/kg)")
    hed_result = 0

    with st.expander("📝 Input Calculation Parameters", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            mouse_dose = st.number_input("Mouse Dose", min_value=0.001, value=12.0, step=0.1, key="mouse_dose")
            mouse_weight = st.number_input("Mouse Weight (g)", min_value=10.0, value=20.0, step=1.0, key="mouse_weight")
        with col_b:
            human_weight = st.number_input("Human Weight (kg)", min_value=1.0, value=60.0, step=1.0, key="human_weight")
            human_height = st.number_input("Human Height (cm)", min_value=100.0, value=170.0, step=1.0, key="human_height")
        if hed_method == "Method 3: Tumor Volume Normalization":
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                mouse_tumor = st.number_input("Mouse Tumor Volume (mm³)", min_value=10.0, value=200.0, step=10.0)
            with col_t2:
                human_tumor = st.number_input("Human Tumor Volume (mm³)", min_value=100.0, value=3000.0, step=100.0)

    calc_hed = st.button("🧮 Calculate HED", type="primary", use_container_width=True)
    if calc_hed:
        mouse_w_kg = mouse_weight / 1000
        if hed_method == "Method 1: Custom BSA Universal Formula":
            human_bsa = 0.007184 * (human_weight ** 0.425) * (human_height ** 0.725)
            mouse_bsa = 0.1 * (mouse_w_kg ** 0.67)
            hed_result = mouse_dose * (mouse_bsa / human_bsa)
        elif hed_method == "Method 2: Allometric Scaling Law (b=0.67)":
            hed_result = mouse_dose * ((mouse_w_kg / human_weight) ** 0.67)
        elif hed_method == "Method 3: Tumor Volume Normalization":
            hed_result = mouse_dose * (mouse_tumor / human_tumor)
        total_injection_mbq = hed_result * human_weight
        hed_record = {
            "Calculation Time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "Method": hed_method.replace("Method 1: ", "").replace("Method 2: ", "").replace("Method 3: ", ""),
            "Mouse Dose(MBq/kg)": round(mouse_dose, 2),
            "Mouse Weight(g)": round(mouse_weight, 1),
            "Human Weight(kg)": round(human_weight, 1),
            "Human Height(cm)": round(human_height, 1),
            "HED(MBq/kg)": round(hed_result, 4),
            "Total Dose(MBq)": round(total_injection_mbq, 4)
        }
        st.session_state.hed_calculation_history.insert(0, hed_record)
        if len(st.session_state.hed_calculation_history) > 10:
            st.session_state.hed_calculation_history.pop()

    st.subheader("📊 HED Results")
    res_hed1, res_hed2, res_hed3 = st.columns(3)
    if calc_hed:
        total_injection_mbq = hed_result * human_weight
        res_hed1.metric("Method", hed_method.replace("Method 1: ", "").replace("Method 2: ", "").replace("Method 3: ", ""))
        res_hed2.metric("Human Equivalent Dose (HED)", f"{hed_result:.4f} MBq/kg")
        res_hed3.metric("Total Recommended Dose", f"{total_injection_mbq:.4f} MBq")
    else:
        st.info("👆 Enter parameters and click [Calculate HED] to get results")

    st.subheader("📜 Recent 10 HED Calculation Records")
    if st.session_state.hed_calculation_history:
        st.dataframe(st.session_state.hed_calculation_history, use_container_width=True, hide_index=True)
    else:
        st.info("No HED calculation records yet")
    st.divider()

    st.markdown("<div style='margin:100px 0'></div>", unsafe_allow_html=True)
    st.subheader("⚙️  Injection Dose Calculation")
    st.divider()
    dose_type = st.selectbox("Select Dose Calculation Type", options=["Fixed Dose", "Weight‑Based Dose"], index=0)
    if dose_type == "Fixed Dose":
        selected_drug = st.selectbox("Select Fixed Dose Drug", options=list(FIXED_DOSE_DRUGS.keys()), index=1)
        preset = FIXED_DOSE_DRUGS[selected_drug]
    else:
        selected_drug = st.selectbox("Select Weight‑Based Drug", options=list(WEIGHT_BASED_DRUGS.keys()), index=0)
        preset = WEIGHT_BASED_DRUGS[selected_drug]

    st.caption("📌 Half‑life Formula: 1/Effective = 1/Physical + 1/Biological")
    col_phys, col_bio, col_eff = st.columns(3)
    with col_phys:
        physical_half_life = st.number_input("Physical Half‑life (days)", min_value=0.1, value=preset["physical_half_life"], step=0.1)
    with col_bio:
        biological_half_life = st.number_input("Biological Half‑life (days)", min_value=0.1, value=20.0, step=0.1)
    with col_eff:
        effective_half_life = 1 / (1 / physical_half_life + 1 / biological_half_life) if physical_half_life > 0 and biological_half_life > 0 else 0.0
        st.number_input("Effective Half‑life (days)", value=round(effective_half_life, 4), disabled=False)

    col3, col4 = st.columns(2)
    with col3:
        weight = st.number_input("Patient Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=1.0)
    with col4:
        time_elapsed = st.number_input("Time Elapsed Since Calibration (days)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)

    col5, col6 = st.columns(2)
    with col5:
        calibration_concentration = st.number_input("Calibration Concentration (MBq/mL)", min_value=0.001, max_value=100000.0, value=preset["calibration_concentration"], step=0.0001, format="%.4f")
    with col6:
        if dose_type == "Fixed Dose":
            total_recommended_activity_fixed = st.number_input("Fixed Total Activity (MBq)", min_value=10.0, value=preset["fixed_total_activity"], step=0.0001, format="%.4f")
        else:
            target_dose_per_kg = st.number_input("Target Dose (MBq/kg)", min_value=0.001, max_value=5.0, value=preset["target_dose_per_kg"], step=0.0001, format="%.4f")

    calc_button = st.button("🧮 Calculate Injection Dose", type="primary", use_container_width=True)
    total_recommended_activity = decay_coefficient = injection_volume = total_activity_gbq = 0.0
    if calc_button:
        if dose_type == "Fixed Dose":
            total_recommended_activity = total_recommended_activity_fixed
        else:
            total_recommended_activity = weight * target_dose_per_kg
        lambda_decay = math.log(2) / physical_half_life
        decay_coefficient = math.exp(-lambda_decay * time_elapsed)
        injection_volume = total_recommended_activity / (decay_coefficient * calibration_concentration)
        total_activity_gbq = total_recommended_activity / 1000
        record = {
            "Calculation Time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "Drug Name": selected_drug,
            "Dose Type": dose_type,
            "Physical Half‑life(d)": round(physical_half_life, 2),
            "Total Activity(MBq)": round(total_recommended_activity, 4),
            "Total Activity(GBq)": round(total_activity_gbq, 4),
            "Decay Coefficient": round(decay_coefficient, 4),
            "Injection Volume(mL)": round(injection_volume, 4)
        }
        st.session_state.calculation_history.insert(0, record)
        if len(st.session_state.calculation_history) > 10:
            st.session_state.calculation_history.pop()

    st.subheader("📊 Injection Dose Results")
    if calc_button:
        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
        res_col1.metric("Drug", selected_drug)
        res_col2.metric("Dose Type", dose_type)
        res_col3.metric("Total Activity(MBq)", f"{total_recommended_activity:.4f}")
        res_col4.metric("Decay Coefficient", f"{decay_coefficient:.4f}")
        res_col5.metric("Injection Volume(mL)", f"{injection_volume:.4f} mL")
    else:
        st.info("👆 Set parameters and click [Calculate Injection Dose] to get results")

    if calc_button and selected_drug == "¹⁷⁷Lu-PSMA-617" and dose_type == "Fixed Dose":
        st.divider()
        st.subheader("🧬 ¹⁷⁷Lu‑PSMA‑617 Organ Absorbed Dose Comparison")
        st.caption(f"📌 Based on Total Activity: {total_activity_gbq:.3f} GBq | Unit: Absorbed Dose(Gy)")
        compare_data = []
        for organ in ORGAN_DOSE_TABLE2.keys():
            human_dose = ORGAN_DOSE_TABLE2[organ] * total_activity_gbq
            mouse_dose = MOUSE_TUMOR_DOSE[organ] * total_activity_gbq
            compare_data.append({
                "Organ": organ,
                "Human(Gy/GBq)": ORGAN_DOSE_TABLE2[organ],
                "Mouse(Gy/GBq)": MOUSE_TUMOR_DOSE[organ],
                "Human Dose(Gy)": round(human_dose, 4),
                "Mouse Dose(Gy)": round(mouse_dose, 4)
            })
        compare_data.append({
            "Organ": "Tumor",
            "Human(Gy/GBq)": "-",
            "Mouse(Gy/GBq)": MOUSE_TUMOR_DOSE["Tumor"],
            "Human Dose(Gy)": "-",
            "Mouse Dose(Gy)": round(MOUSE_TUMOR_DOSE["Tumor"] * total_activity_gbq, 4)
        })
        df = pd.DataFrame(compare_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("<div style='margin:60px 0'></div>", unsafe_allow_html=True)
    st.subheader("⏱️ In Vivo Residual Activity Calculation")
    st.divider()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        post_injection_time = st.number_input("Time After Injection", min_value=0.0, value=24.0, step=1.0)
        time_unit = st.selectbox("Time Unit", ["Hours (h)", "Days (d)"], index=0)
    t_h = post_injection_time * 24 if time_unit == "Days (d)" else post_injection_time
    if calc_button and total_recommended_activity > 0:
        lambda_eff = math.log(2) / effective_half_life
        remaining_activity_mbq = total_recommended_activity * math.exp(-lambda_eff * (t_h / 24))
        remaining_percent = remaining_activity_mbq / total_recommended_activity * 100
        st.subheader("📈 Residual Activity Results")
        rem1, rem2, rem3, rem4 = st.columns(4)
        rem1.metric("Time After Injection", f"{post_injection_time} {time_unit}")
        rem2.metric("Initial Total Activity", f"{total_recommended_activity:.4f} MBq")
        rem3.metric("Residual Activity", f"{remaining_activity_mbq:.4f} MBq")
        rem4.metric("Residual Ratio", f"{remaining_percent:.4f} %")
    else:
        st.info("👆 Complete Step 1 first to calculate residual activity")

    st.divider()
    st.subheader("📜 Recent 10 Injection Dose Records")
    if st.session_state.calculation_history:
        st.dataframe(st.session_state.calculation_history, use_container_width=True, hide_index=True)
    else:
        st.info("No injection calculation records yet")
#公式参考
    # st.divider()
    # st.subheader("📐 Calculation Formulas")
    # st.markdown("##### 🧪 HED Extrapolation Formulas")
    # st.latex(r"HED = D_{mouse} \times \dfrac{BSA_{mouse}}{BSA_{human}}")
    # st.latex(r"HED = D_{mouse} \times \left( \dfrac{W_{mouse}}{W_{human}} \right)^{0.67}")
    # st.latex(r"HED = D_{mouse} \times \dfrac{TV_{mouse}}{TV_{human}}")
    # st.divider()
    # st.markdown("##### ⚛️ Half‑life & Injection Dose Formulas")
    # st.markdown("**Half‑life Relationship**")
    # st.latex(r"\frac{1}{t_{eff}} = \frac{1}{t_{phys}} + \frac{1}{t_{bio}}")
    # st.markdown("**Decay Coefficient & Injection Volume**")
    # st.latex(r"k = e^{-\frac{\ln2}{t_{phys}} \times t}")
    # st.latex(r"\text{Injection Volume(mL)} = \frac{\text{Total Activity(MBq)}}{k \times \text{Conc.(MBq/mL)}}")
    # st.markdown("**Residual Activity in Vivo**")
    # st.latex(r"A_t(\text{MBq}) = A_0(\text{MBq}) \cdot e^{-\frac{\ln2}{t_{eff}} \cdot t}")
#使用安全警告
    st.divider()
    st.warning("""
    ⚠️ Radiation Safety & Academic‑Use Disclaimer
    1. This calculator serves academic research, pre‑clinical studies and clinical dosimetry assistance only. It shall not be the sole basis for radiopharmaceutical administration; final clinical decisions must be determined by qualified nuclear medicine physicians.
    2. Physical half‑life is an intrinsic radionuclide constant, while biological half‑life is an empirical reference affected by individual metabolism, pathology and organ function.
    3. Preset radiopharmaceutical doses are literature‑based references. Clinical application must comply with official drug specifications, clinical guidelines and institutional standards.
    4. Total activity, injection volume, decay correction and human equivalent dose (HED) shall be verified before clinical administration following radiation protection protocols.
    5. HED extrapolation, organ‑specific dose and residual activity are theoretical approximations, without accounting for tumour heterogeneity, radiosensitivity and inter‑species physiological differences.
    6. Radiopharmaceuticals emit ionising radiation. All relevant operations shall follow national radiological regulations to protect patients, medical staff and the public.
    7. Decay correction is calculated via pure physical kinetics, excluding biological clearance, non‑specific binding and excretion, thus involving inherent computational uncertainties.
    """)
