# Data Science Programmentwurf

A university exam project (*Programmentwurf*) for a Data Science course, applying the full **CRISP-DM** process to a fictional real estate dataset. The notebook walks through business understanding, exploratory data analysis, data preparation, regression modeling, model comparison/stacking, deployment recommendations, and unsupervised clustering.

## Business Context

The fictional client runs a door-to-door marketing business for craft and home-improvement services (painting, door repair, new awnings, garden landscaping, chimney sweeping, pool construction, full renovations). The underlying assumption is that owners of more expensive homes are willing to spend more on these services. The goal of the analysis is to understand which house features drive sale price, so the client can target its marketing campaigns toward the most profitable households.

## Repository Structure

```
.
├── Programmentwurf/
│   ├── programmentwurf.ipynb      # Main analysis notebook (all 7 tasks)
│   ├── data_for_training.csv      # Training dataset (~2,385 rows, ";"-delimited)
│   ├── data_for_test.csv          # Holdout dataset for final predictions
│   └── data_for_test_filled.csv   # Test dataset after cleaning/imputation
├── Pruefungsleistung2023_1_v1.0.pdf  # Original assignment specification
├── output*.png / training_curve.PNG  # Exported plots referenced in the notebook
├── cheatsheet.url                 # Link to a scikit-learn cheat sheet used as reference
└── README.md
```

## Dataset

The dataset describes residential properties with features grouped into four categories:

- **General attributes:** location (`Lage`), lot size (`Grundstueck_qm`), tax assessment (`Steuerzuordnung`)
- **Quality & condition:** exterior quality/condition, overall quality/condition, kitchen condition, fireplace quality, heating condition
- **Construction & renovation:** year built, garage year built, renovation year, sale year
- **Additional features:** special features (e.g. pool, family sale), garage spaces, number of fireplaces, basement type

The target variable is the sale price (`Zz_Verkaufspreis`).

## CRISP-DM Workflow (Notebook Tasks)

1. **Business Understanding** -> defines the business goals, success criteria, and the key questions: which features correlate strongly with each other and which most/least influence sale price.
2. **Data Exploration & Analysis** -> descriptive statistics, missing-value analysis, distribution and skewness checks, correlation analysis (cardinal, ordinal, nominal features) and pairplots/heatmaps segmented by location and basement type.
3. **Data Preparation** -> ordinal encoding, log-transformation of skewed features, missing-value imputation strategies per feature, removal of irrelevant columns, one-hot encoding of nominal features, and robust scaling.
4. **Modeling -> Regression with Inference** -> a Lasso regression model (with hyperparameter tuning and feature selection) used both for prediction and for interpretable feature-importance analysis.
5. **Modeling -> Model Comparison** -> Lasso, Random Forest, and Gradient Boosted Trees are tuned and validated, then combined into a `StackingRegressor`. Models are compared via cross-validation, learning curves, and residual analysis; the stacking model is selected as the final model.
6. **Deployment** -> a two-page, stakeholder-facing summary translating the modeling results into concrete marketing recommendations for the client's management and marketing team.
7. **Unsupervised Learning** -> K-Means clustering with the elbow method to determine the optimal number of clusters, visualized via PCA and t-SNE (with and without location as a feature) to segment properties.

## Key Results

- The final **Stacking Regressor** (Lasso + Random Forest + Gradient Boosted Trees) achieved the best validation performance among all candidate models, with a high R² and low overfitting gap between training and validation sets.
- The features with the strongest influence on sale price were **pool ownership**, **living area**, **location (Heroes Park)**, and **number of fireplaces**.
- Features with **little to no influence** on sale price included tax assessment, overall condition, and exterior condition. These were dropped during data preparation.
- The optimal number of clusters for property segmentation, determined via the elbow method, was **3**.

## Tech Stack

- Python 3 (Jupyter Notebook)
- `pandas`, `numpy` -> data handling
- `matplotlib`, `seaborn` -> visualization
- `scikit-learn` -> preprocessing (`OrdinalEncoder`, `RobustScaler`, `FunctionTransformer`), modeling (`Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`, `StackingRegressor`, `TransformedTargetRegressor`), model selection (`GridSearchCV`, `RandomizedSearchCV`, `cross_validate`, `learning_curve`), and dimensionality reduction/clustering (`PCA`, `TSNE`, `KMeans`)
- `scipy.stats` -> distributions used in randomized hyperparameter search

## Getting Started

```bash
git clone https://github.com/juliabb1/Data-Science-Programmentwurf.git
cd Data-Science-Programmentwurf/Programmentwurf
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook programmentwurf.ipynb
```

Run the notebook top to bottom; it reads `data_for_training.csv` for model development and `data_for_test.csv` for the final holdout predictions.

## Notes

- This project was created as an academic exam submission and follows the brief in `Pruefungsleistung2023_1_v1.0.pdf`.
- All data is fictional and was provided as part of the coursework; it does not represent real properties or sale prices.
- The notebook's analysis, markdown commentary, and final recommendations are written in German, per the course requirements.
