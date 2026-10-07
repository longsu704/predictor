# Radiopharmaceutical Biodistribution Prediction Platform

A Streamlit web application for predicting radiopharmaceutical tumour uptake (% ID/g),  
with an integrated module for dose extrapolation and injection-dose calculation.

**Live demo:** <https://radiopharmaceutical-prediction-platform.streamlit.app/>

---

## How to Run

### Option 1 — Use the online app

Open <https://radiopharmaceutical-prediction-platform.streamlit.app/>. Nothing to install.

### Option 2 — Run locally

You need **Python 3.11** and the six model files in the same folder as the script.

1. Download or clone this repository to a local folder. Keep every file in the  
   same folder — do not move `predictor1_prep.py` or the `.pkl` files into subfolders.
2. Open a terminal (on Windows: Command Prompt or PowerShell) and go to that folder:
   ```bash
   cd path/to/the/folder
   ```
3. Install the dependencies (only needed the first time):
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run predictor1.py
   ```
5. Your browser opens automatically at <http://localhost:8501>. If it does not,  
   paste that address into your browser manually. To stop the app, press  
   `Ctrl + C` in the terminal.

### Option 3 — Deploy your own copy

1. Push this repository to GitHub. Upload **all files to the repository root** —  
   the flat layout matters, because the app looks for the model files in its own folder.
2. Sign in at <https://share.streamlit.io> using your GitHub account.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository**: `your-username/predictor`
   - **Branch**: `main`
   - **Main file path**: `predictor1.py`
5. (Optional) Open **Advanced settings** to select the Python version.
6. Click **Deploy**. The first build takes a few minutes; later updates rebuild  
   automatically whenever you push a commit.

> **If the app reports `Model file not found`:** the six `.pkl` files are missing from  
> the repository, sit in a subfolder, or were skipped by `.gitignore`. They must be  
> committed at the repository root, next to `predictor1.py`.

---

## What the Code Does

- **Two models in one app.** *CFM* (Core Feature Model) uses 12 inputs;  
  *EFM* (Extended Feature Model) uses 13 — it adds the chelating agent as an extra input.
- **Three validation schemes.** Every model exists in three variants: `Random`,  
  `Compound` (leave-compound-out) and `Group` (leave-publication-out). The latter two  
  answer "how well does this generalise to an unseen compound / publication?".
- **Self-contained model files.** All preprocessing — target encoding of categorical  
  variables, median imputation and standardisation — is packaged *inside* each `.pkl`.  
  Loading a model therefore needs no separate feature-engineering step, and the  
  prediction path can never drift from the training path.
- **Log-space training.** The model learns `log1p(% ID/g)` and converts back with  
  `expm1` for display, so the reported value stays on the same scale as the  
  decision thresholds.
- **Input values persist.** Switching between models — including CFM ↔ EFM — keeps  
  every field you have already filled in, chelating agent included.
- **No external data files at runtime.** All dropdown options and dose tables are  
  defined in the source code, so the app only needs the code plus the six `.pkl` files.

---

## Using the Interface

The left sidebar (**Function Navigation**) switches between the two modules.

### Module 1 — Radiopharmaceutical Biodistribution Prediction

1. **1. Model Selection** — choose the feature set (*CFM* or *EFM*) and the  
   validation scheme (*Random*, *Compound*, *Group*).
2. **2. Input Parameters** — fill in six categorical fields from the dropdowns  
   (Nuclide, Target, Nuclide Ray Properties, Tumor Cell, Mouse Species, Tumor Models,  
   plus Chelating Agent for EFM) and six numerical fields (Kd, IC50, Molecular Weight,  
   LogD, Injection Dosage, Time). **Leave a numerical field as `NA` if the value is  
   unknown** — it will be imputed automatically.
3. **3. Prediction & Results** — click **Run Prediction**. You get a summary of your  
   input, the predicted uptake in % ID/g (with the underlying log-scale value in  
   brackets), and the model and split scheme used.
4. **PSMA-617 evaluation.** If, and only if, your input is `177Lu` + `PSMA` at 1, 4 or  
   24 h (plus `DOTA` for EFM), the app compares your prediction against the published  
   PSMA-617 standard and, when it falls short, searches for the injected dose that  
   would reach it. The search is listed as a 20-row table from 0.5 to 10 in steps of 0.5.
5. **4. Recent Prediction History (Last 10)** — your last ten predictions, with a  
   **Clear History** button.

### Module 2 — Radioactive Drug Calculator

1. **HED extrapolation** — pick one of three methods (custom BSA formula, allometric  
   scaling with b = 0.67, or tumour-volume normalisation), enter the mouse dose and  
   body weight and the human weight and height, then click **Calculate HED**.
2. **Injection dose calculation** — choose *Fixed Dose* or *Weight-Based Dose*, select  
   the radiopharmaceutical, fill in the remaining parameters, and click  
   **Calculate Injection Dose**.
3. **¹⁷⁷Lu-PSMA-617 organ absorbed-dose comparison** — human organ doses for reference.
4. **In vivo residual activity** — enter the starting activity and the elapsed time  
   (switch between hours and days), and the decayed activity is calculated.
5. HED and injection-dose runs are each kept in their own ten-entry history list.

---

## Data

- `Biodistribution_data_radiopharmaceuticals.xlsx` — **processed** dataset: merged cells  
  removed and filled, and rows with missing tumour values dropped. This is the table the  
  deployed models were trained on.
- `Biodistribution data of radiopharmaceuticals.xlsx` — **unprocessed original** dataset:  
  merged cells intact, includes references.

---

## Note

This tool is intended for research use only and must not be used for clinical decision-making.
