# Crop Disease Detection using Machine Learning

A supervised multi-class classification project that predicts crop disease type (Healthy, Fungal, Bacterial, or Pest Damage) from IoT sensor, soil, and farm management data — built as a 3-phase capstone project covering EDA, data cleaning & feature engineering, and model building.

## Problem Statement

Agriculture is central to India's economy, and crop disease is one of the biggest threats to productivity. This project simulates the role of a junior data scientist at an AgriTech startup tasked with building a model that classifies farm observations — collected via IoT sensors across partner farms — into one of four disease categories, enabling timely intervention (fungicide, bactericide, pesticide, or routine monitoring).

**Dataset:** 5,000 rows × 18 columns (17 features + 1 target: `disease_label`)
Intentionally messy — containing missing values, inconsistent categorical capitalisation, and physically implausible outliers (e.g., negative fertilizer values, temperature spikes >100°C), requiring hands-on data cleaning before modeling.

## Project Structure

```
├── notebooks/
│   ├── 1_CropDisease-EDA.ipynb                  # Phase 1: Exploratory Data Analysis
│   ├── 2_CropDisease-DataCleaning-FE.ipynb      # Phase 2: Data Cleaning & Feature Engineering
│   └── 3_Crop_Disease_Modelling.ipynb           # Phase 3: Model Building & Evaluation
├── data/
│   ├── Original_Dataset.csv                     # Raw, uncleaned dataset
│   └── Cleaned_Dataset.csv                      # Cleaned & feature-engineered dataset
├── docs/
│   ├── Problem_Statement.pdf                    # Capstone brief
│   └── Model_Comparison.docx                    # Final model comparison & recommendation
└── README.md
```

## Approach

### Phase 1 — Exploratory Data Analysis
- Inspected shape, dtypes, and missing values (highest in categorical columns: `season` 28.1%, `irrigation_method` 24.3%, `soil_type` 22.5%, `region` 21.1%)
- Detected inconsistent capitalisation across all categorical columns (e.g., `WHEAT` / `Wheat` / `wheat`)
- Checked class balance: Healthy (30.6%), Bacterial (25.1%), Fungal (24.3%), Pest Damage (20%) — mild imbalance
- Identified skewed features (`rainfall_mm` skew ≈ 9.9) and physically implausible outliers via histograms and boxplots
- Built a correlation heatmap — numerical features showed largely weak linear correlation with each other
- Compared key features (humidity, leaf wetness) across disease classes to surface early discriminating patterns

### Phase 2 — Data Cleaning & Feature Engineering
- Standardised all categorical columns to Title Case
- Replaced physically impossible values (negative temperature, humidity >100%, negative fertilizer, rainfall ≈999mm/week) with NaN using domain-driven bounds
- Imputed missing values: **mean** for symmetric numerical columns, **median** for skewed ones, **mode** for categorical columns — retaining all 5,000 rows
- Engineered new features:
  - `HTI` (Heat-Humidity stress Index) — combines temperature and humidity into a single stress signal for fungal/bacterial risk
  - `drought_stress` — binary flag when `days_since_last_rain > 20` and `rainfall_mm < 10`
  - `maturity_stage` — plant growth stage (Seedling / Vegetative / Reproductive) derived from plant age and label-encoded due to its natural order
- One-Hot Encoded remaining nominal columns (`crop_type`, `region`, `season`, `soil_type`, `irrigation_method`), expanding to ~38 columns

### Phase 3 — Model Building & Evaluation
- 80/20 stratified train-test split to preserve class proportions
- Trained and compared three models: **Decision Tree**, **KNN** (k=3, 5, 7), and **Random Forest**
- Evaluated using Test Accuracy and **Macro F1-score** (chosen over plain accuracy to fairly weight the minority `Pest Damage` class)
- Analyzed confusion matrix and Random Forest feature importances — `HTI` (engineered) and `leaf_wetness_hrs` ranked among the top predictors

## Model Comparison

| Model | Train Acc | Test Acc | Macro F1 | Verdict |
|---|---|---|---|---|
| Decision Tree | 1.000 | 0.607 | 0.601 | Overfits (large train-test gap) |
| KNN (k=3) | 0.695 | 0.389 | 0.384 | Poor |
| KNN (k=5) | 0.639 | 0.402 | 0.390 | Poor |
| KNN (k=7) | 0.621 | 0.404 | 0.399 | Best KNN, still weak |
| **Random Forest** | 1.000 | **0.677** | **0.661** | ✅ Recommended |

## Final Recommendation

**Random Forest Classifier** (`n_estimators=100`, `class_weight='balanced'`) is recommended for deployment. It achieved the highest test accuracy (67.7%) and Macro F1 (0.661) — outperforming the Decision Tree by 7 points and the best KNN variant by 26 points. The ensemble approach resolves the single Decision Tree's overfitting, while `class_weight='balanced'` ensured the minority `Pest Damage` class (20% of records) wasn't ignored during training, lifting its F1-score to 0.67.

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## How to Run

```bash
git clone https://github.com/akellapadmaja/Crop-Disease-Detection.git
cd Crop-Disease-Detection
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Run the notebooks in order: `1_CropDisease-EDA.ipynb` → `2_CropDisease-DataCleaning-FE.ipynb` → `3_Crop_Disease_Modelling.ipynb`
