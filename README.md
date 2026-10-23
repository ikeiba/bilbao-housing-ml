# 🏡 Bilbao Housing Market Machine Learning (`bilbao-housing-ml`)

[![Spanish Version](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](README.es.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-008080)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org/)

> **Language switch:** [🇪🇸 Leer en Español](README.es.md) | **[🇬🇧 English Version](README.md)**

---

## 📌 Executive Summary

**bilbao-housing-ml** is an end-to-end Machine Learning research project that analyzes, models, and segments the residential real estate market in **Bilbao (Bizkaia, Spain)**. 

Beginning with autonomous web scraping from major real estate portals (Idealista), the project addresses common challenges of real-world housing data: extreme sparsity, class imbalance, non-linear valuation dynamics, and noisy unstructured text. Through advanced multi-model imputation, Bayesian hyperparameter optimization (Optuna), and model explainability techniques (SHAP and ALE), the repository covers:
1. **Accurate Price Valuation (Regression):** Predicting listing prices using non-linear gradient-boosted trees (CatBoost) achieving a **~15.0% MAPE** and **$R^2 \approx 0.86$**.
2. **Urban District Inference (Classification):** Classifying properties into Bilbao's 11 municipal zones using LightGBM with engineered spatial-financial indicators, reaching **70% multi-class accuracy**.
3. **Housing Typology Segmentation (Clustering):** Uncovering natural real estate clusters through a hybrid **DBSCAN + KNN** density pipeline, revealing 5 core archetypes and an exclusive luxury tier.

---

## 📁 Repository Structure

```text
bilbao-housing-ml/
│
├── classification/
│   ├── classification_zona.ipynb      # Multi-class zone classification (LightGBM, Optuna, SHAP, ALE)
│   └── classification_estado.ipynb    # [WIP] Property condition classification (good condition vs to renovate)
│
├── cleaning/
│   ├── catboost_info/                 # Imputation model training checkpoints
│   └── data_cleaning.ipynb            # End-to-end preprocessing, feature engineering & ML-based imputation
│
├── clustering/
│   └── clustering.ipynb               # Unsupervised market segmentation (KMeans, GMM, BIRCH, DBSCAN + KNN)
│
├── data/
│   ├── raw/
│   │   ├── idealista_urls.csv         # Scraped property URLs across 11 Bilbao districts (1,489 rows)
│   │   └── idealista_viviendas_detalle.csv # Raw scraped property attributes (1,490 rows x 25 features)
│   └── cleaned/
│       └── data_final.csv             # Final preprocessed and imputed dataset (1,229 rows x 21 features)
│
├── deep_learning/
│   └── deep_learning.ipynb            # [Roadmap] NLP & embeddings from property descriptions
│
├── regression/
│   ├── catboost_info/                 # Model artifacts and logs
│   └── regression.ipynb               # Price regression pipeline (EDA, log-transform, Optuna, CatBoost, SHAP/ALE)
│
├── scrapers/
│   └── house_info_scraper.ipynb       # Two-tier scraper using undetected-chromedriver & BeautifulSoup
│
├── README.md                          # Main project documentation (English)
└── README.es.md                       # Spanish version of documentation
```

---

## 🕷️ 1. Data Acquisition (Web Scraping)

The raw data was gathered directly from **Idealista** using a custom two-phase scraping pipeline located in [`scrapers/house_info_scraper.ipynb`](scrapers/house_info_scraper.ipynb):

1. **URL Discovery Stage:**
   - Navigates through all 11 official Bilbao districts: *Deusto, Uribarri, Otxarkoaga-Txurdinaga, Basurto-Zorroza, Abando-Albia, Indautxu, Casco Viejo, Begoña-Santutxu, Rekalde, Ibaiondo, and San Adrián - La Peña*.
   - Uses `undetected_chromedriver` and `selenium` to paginate search listings, outputting **1,489 unique listing URLs** saved in [`data/raw/idealista_urls.csv`](data/raw/idealista_urls.csv).
2. **Property Details Extraction Stage:**
   - Visits each individual listing page and parses HTML payloads using `BeautifulSoup`.
   - Extracts 25 raw features: price, zone, neighborhood, built area ($m^2$), usable area ($m^2$), bedrooms, bathrooms, floor level, interior/exterior, elevator, garage, storage room, balcony, new construction vs. second hand, condition, construction year, heating, professional agency vs. private seller, energy consumption and emissions (values and letter grades A-G), and full textual descriptions.
   - Saves results incrementally to [`data/raw/idealista_viviendas_detalle.csv`](data/raw/idealista_viviendas_detalle.csv) with error logging and ban recovery mechanisms.

---

## 🧹 2. Data Cleaning & ML-Based Imputation

Raw real-estate web data contains significant missingness and formatting inconsistencies. Rather than dropping rows or using naive global mean imputation, the team implemented domain-specific and machine learning imputation pipelines in [`cleaning/data_cleaning.ipynb`](cleaning/data_cleaning.ipynb):

* **General Preprocessing & Deduplication:** Removed duplicates, formatted column names to English snake_case, stripped currency symbols, and cast numeric columns.
* **Usable Area (`usable_area`):** Compared neighborhood/zone mean ratios against **Zone-Specific Linear Regression** models predicting usable square meters from built square meters ($m^2$), achieving robust interpolation without violating dimensional logic ($usable \le built$).
* **Condition (`condition`):** Identified that listings with null condition coincided with new developments; imputed as `"Nuevo"` (*New Construction*).
* **Construction Year (`year`):** 479 non-new properties lacked construction year. Tested a benchmark ladder:
  - Global Lasso Regression vs. Lasso + RANSAC (outlier-robust).
  - Zone-specific Lasso vs. Zone-specific RANSAC.
  - Multi-tree ensembles (**CatBoostRegressor** and **RandomForestRegressor**).
  - Implemented the champion model per zone (and global CatBoost/RandomForest where zone samples were limited) to preserve historical building age distributions.
* **Exterior Orientation (`exterior`):** Extracted keywords (`exterior`, `terraza`, `balcon`) from unstructured textual descriptions. Remaining missing values were predicted using an **SVM classification pipeline** with tuned classification thresholds ($\tau \approx 0.2179$) to properly reflect base class proportions.
* **Floor Level (`floor`):** Standardized qualitative floor strings into an ordinal numeric scale: `-2` (Basements / Semisótanos), `-1` (Single-family Chalets / Houses), `0` (Ground floors / Bajos / Entreplantas), and `1, 2, ...` (standard building floors).
* **Energy Ratings (`emissions_label`, `consumption_label`):** Evaluated Random Forest, Logistic Regression, Naive Bayes, and SVM with Stratified 5-Fold Cross-Validation. A **Random Forest Classifier** was retained as the champion model to impute energy efficiency grades ($A$ through $G$) across the missing records.
* **Feature Pruning:** Dropped overly sparse columns (`heating`, raw numeric emission/consumption indices) while preserving high-confidence categories.
* **Final Cleaned Dataset:** **1,229 properties** and **21 clean attributes** saved to [`data/cleaned/data_final.csv`](data/cleaned/data_final.csv).

---

## 📈 3. Price Regression Pipeline

Implemented in [`regression/regression.ipynb`](regression/regression.ipynb) to model and predict housing asking prices.

### Exploratory Data Analysis & Transformations
- Target variable `price` exhibited strong right skewness and heteroscedasticity. Applying a natural logarithmic transform ($\log(\text{price})$) normalized the target distribution and stabilized residual variances.
- Evaluated leverage points and collinearity between `built_area`, `usable_area`, and room counts.

### Model Benchmarking & Hyperparameter Optimization
Evaluated multiple model families using cross-validation:
1. **Ordinary Least Squares (OLS) Linear Regression:** Baseline comparison without transformations vs. with log-transform ($R^2 \approx 0.858$, $\text{MAPE} \approx 17.2\%$).
2. **Regularized Linear Models:** Ridge, Lasso, and ElasticNet tuned via Optuna ($\text{MAPE} \approx 17.01\%$, $R^2 \approx 0.856$).
3. **Non-linear Ensemble Models:** Evaluated Random Forest, XGBoost, LightGBM, and CatBoost.

### Champion Regression Model: CatBoost Regressor
**CatBoost** outperformed all other architectures. Hyperparameters were tuned with **Optuna** (minimizing MAPE across validation folds):
- **Optimal Hyperparameters:** `iterations`: 2758, `depth`: 6, `learning_rate`: 0.044, `l2_leaf_reg`: 10.29, `bagging_temperature`: 0.659, `random_strength`: 9.49, `border_count`: 164.

| Metric | Cross-Validation / Validation | Holdout Test Set | Assessment |
| :--- | :---: | :---: | :--- |
| **MAPE (%)** | **15.08%** | **15.70%** | Exceptional precision for open-market real estate data |
| **$R^2$ Score** | **0.8646** | **0.8005** | High explained variance without overfitting |
| **MAE (€)** | **~€68,160** | **~€72,400** | Stable generalization across unseen price tiers |

### Explainability & Interpretability
- **Feature Importance:** Bathrooms, usable area, and built area were determined to be the primary valuation drivers; construction year and floor provided secondary refinements.
- **Accumulated Local Effects (ALE):** Verified monotonic, non-linear returns on usable square meters and floor levels.
- **SHAP (SHapley Additive exPlanations):**
  - **Local Waterfall Plots:** Detailed Euro-scale explanations showing exactly which attributes increased or decreased the estimated price of individual properties.
  - **Global SHAP Summary / Beeswarm Plots:** Confirmed the magnitude and directionality of each feature on log-price across the entire dataset.

---

## 🗺️ 4. Zone Multi-Class Classification

Implemented in [`classification/classification_zona.ipynb`](classification/classification_zona.ipynb) to classify listings into Bilbao's **11 municipal zones** purely from property attributes (without using neighborhood or coordinates).

### Methodology & Iterative Experiments
- **Class Imbalance:** Zone frequencies ranged from 189 listings in dense core districts (*Abando-Albia*) down to 28 listings in smaller districts (*San Adrián - La Peña*, *Otxarkoaga*). Evaluated via Stratified 5-Fold Cross-Validation monitoring Accuracy, Balanced Accuracy, Weighted/Macro F1, and Matthews Correlation Coefficient.
- **Tested Strategies:**
  - *Strategy 1:* One-Hot Encoding categorical features on standard algorithms.
  - *Strategy 1.1:* Native categorical handling using XGBoost, LightGBM, and CatBoost.
  - *Strategy 1.2:* Reduced-dimension Support Vector Machines (Linear & RBF).
  - *Strategy 1.3:* Inverse class-weight balancing (`class_weight='balanced'`).
  - *Strategy 1.4 (Feature Engineering):* Introduced `price_m2 = price / built_area`, which provided a critical price-density discriminator between historical central areas and peripheral districts.
  - *Strategy 1.5:* Bayesian Hyperparameter Optimization with **Optuna** over 50 trials tuning tree depth, leaf counts, learning rates, subsampling, and regularizations.

### Champion Classification Model: Tuned LightGBM
- **Test Accuracy:** **70.0%**
- **Weighted F1-Score:** **0.70** (Precision: 0.71, Recall: 0.70)
- **Macro F1-Score:** **0.69** (demonstrating strong accuracy even on minority zones such as Otxarkoaga-Txurdinaga with F1 = 0.86)

```text
                       precision    recall  f1-score   support
         abando-albia       0.75      0.79      0.77        48
      basurto-zorroza       0.70      0.64      0.67        25
      begona-santutxu       0.58      0.73      0.65        15
          casco-viejo       0.88      0.58      0.70        12
               deusto       0.58      0.83      0.68        18
             ibaiondo       0.68      0.68      0.68        31
             indautxu       0.86      0.70      0.78        27
otxarkoaga-txurdinaga       0.86      0.86      0.86         7
              rekalde       0.67      0.61      0.63        33
   san-adrian-la-pena       0.50      0.43      0.46         7
             uribarri       0.70      0.70      0.70        23

             accuracy                           0.70       246
            macro avg       0.70      0.69      0.69       246
         weighted avg       0.71      0.70      0.70       246
```

- **Interpretability:** Model decisions were analyzed through ALE curves on top numerical features (`year`, `price_m2`, `price`, `usable_area`) and multi-class SHAP attributions.

---

## 🧩 5. Unsupervised Market Segmentation (Clustering)

Implemented in [`clustering/clustering.ipynb`](clustering/clustering.ipynb) to discover natural property typologies across Bilbao without ground-truth labels.

### Algorithmic Evaluation
- Scaled numerical features: `price`, `built_area`, `usable_area`, `bedrooms`, `bathrooms`, `floor`, and `year`.
- Evaluated **K-Means** (Elbow method, Silhouette scores 0.36 – 0.57), **Gaussian Mixture Models (GMM)** with BIC/AIC minimum criteria, and **BIRCH** tree clustering.
- Selected **DBSCAN** for its ability to find arbitrary density-connected clusters without forcing spherical shapes or uniform sizes.

### Outlier Handling via KNN Reassignment
- DBSCAN grid search isolated dense clusters but left 579 boundary/noise instances.
- Developed an automated **KNN (K-Nearest Neighbors) distance evaluation** to compute optimal neighborhood reachability.
- Applied a strict cutoff distance threshold ($\text{distance} \le 5.50$):
  - Boundary instances were smoothly reassigned to their closest natural cluster.
  - The 29 genuine extreme outliers were preserved in an explicit **Luxury / Giant Estate Cluster (`-1`)**.

### Discovered Housing Archetypes

| Cluster | Count | Mean Price | Mean Built Area | Beds / Baths | Mean Year | Distinctive Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`-1`** | 29 | €1,451,102 | 377.0 $m^2$ | 7.7 / 3.4 | 1938 | **Luxury Mansions & Penthouses:** Historical prestige properties concentrated in Abando-Albia (51.7%) and Indautxu (20.7%). |
| **`0`** | 110 | €273,375 | 56.5 $m^2$ | 1.0 / 1.0 | 1960 | **Urban Studios & 1-Bed Flats:** Compact living with 100% elevator access, popular among young professionals and singles. |
| **`1`** | 92 | €415,208 | 88.8 $m^2$ | 2.0 / 2.0 | 1997 | **Modern / New Developments:** 47.8% new construction, 46.7% Energy Label A, 44.6% garage inclusion, 2 full bathrooms. |
| **`2`** | 615 | €611,054 | 140.1 $m^2$ | 3.7 / 2.3 | 1965 | **Prime Family Residences:** The core volume segment; large family flats in prime central areas with balconies (63.4%). |
| **`3`** | 192 | €269,890 | 85.1 $m^2$ | 3.2 / 1.0 | 1952 | **Affordable 3-Bed Flats:** Traditional family flats in peripheral/working-class zones (Ibaiondo, Santutxu, Rekalde); rare garage access (6.2%). |
| **`4`** | 191 | €252,653 | 68.7 $m^2$ | 2.0 / 1.0 | 1948 | **Budget Compact 2-Bed Flats:** Entry-level older apartments in established working-class neighborhoods; lowest average price. |

---

## 🔮 6. Work in Progress & Roadmap

- [ ] **`classification/classification_estado.ipynb`:** Developing binary classification models to predict whether a property is in move-in condition (*buen estado*) vs. requiring major renovation (*a reformar*) based on age, floor level, and pricing discrepancies.
- [ ] **`deep_learning/deep_learning.ipynb`:** Natural Language Processing (NLP) pipeline on property listing descriptions (`description`):
  - Keyword sentiment and premium term extraction (e.g., "vistas despejadas", "luminoso", "reformado de lujo").
  - Pre-trained transformer embeddings (e.g., multilingual RoBERTa / BERT) combined with tabular features for multi-modal price prediction.

---

## 🛠️ Tech Stack & Requirements

- **Python Environment:** Python 3.10+
- **Data Manipulation & Math:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn`, `imbalanced-learn`
- **Gradient Boosting Frameworks:** `catboost`, `lightgbm`, `xgboost`
- **Hyperparameter Optimization:** `optuna`
- **Explainable AI (XAI):** `shap`, `PyALE`
- **Web Scraping:** `selenium`, `undetected-chromedriver`, `beautifulsoup4`

---

## 👥 Contributors & Roles

This project was developed for the Machine Learning course at the **University of Deusto** (Academic Year 2024–2025):

- **David Rodríguez** ([@Davidrs05](https://github.com/Davidrs05)) — Price Regression Pipeline, Year Imputation Regression, Optuna & SHAP evaluation.
- **Enetz Quintana** ([@EnetzUni](https://github.com/EnetzUni)) — Usable Area Imputation Analysis, Deep Learning & NLP Roadmap.
- **Gonzalo Iglesias** ([@gonzaloiglesias](https://github.com/gonzaloiglesias)) — Market Clustering Pipeline (DBSCAN + KNN Outlier Refinement), Exterior Imputation.
- **Iker Ibarrola** ([@ikeiba](https://github.com/ikeiba)) — Scraper Architecture, Multi-class Zone Classification (LightGBM, Optuna, ALE/SHAP), Energy Label Classification.
- **Xabier Villa** ([@XabiVii](https://github.com/XabiVii)) — Floor Level Parsing, Emission/Consumption Imputation, Property Condition Roadmap.

---

## 📄 License & Academic Note

This repository is maintained for educational and research purposes under the Computer Engineering / Data Science degree curriculum at the University of Deusto. Data scraped from publicly available listings is utilized strictly for academic research and benchmarking.
