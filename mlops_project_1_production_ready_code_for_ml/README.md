# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- The purpose of this project is to use production grade skills in SWE to refactor and 'productionise' Machine Learning models previously written in Python on a Jupyter notebook into modular clean code ready for production.
- The main skills used during the refactoring process were:
  - The KISS methodology (Keep It Simple Stupid!) writing easy to read code, reducing repetitive code and increasing performance using vectorised implementations where applicable 
  - Testing techniques with pytest in order to build production grade tested code
  - Linted code with pylint to ensure PEP 8 standard is enforced to maintain elegant and ease to read code.

## Files structure and data description
The root directory is composed by 4 folders with model outputs and 4 main files in the root directory, as below.

 .
`├── `**`data\`**\
`│   └── bank_data.csv`\
`├── `**`images\`**\
`│   ├── `**`eda\`**\
`│   │   ├── churn_distribution.png`\
`│   │   ├── customer_age_distribution.png`\
`│   │   ├── heatmap.png`\
`│   │   ├── marital_status_distribution.png`\
`│   │   └── total_trans_ct_distribution.png`\
`│   └── `**`results\`**\
`│       ├── feature_importances.png`\
`│       ├── logistic_results.png`\
`│       ├── rf_results.png`\
`│       ├── roc_curve_result.png`\
`│       └── shap_explainer_result.png`\
`├── `**`logs\`**\
`├── `**`models\`**\
`│   ├── logistic_model.pkl`\
`│   └── rfc_model.pkl`\
`├── README.md`\
`├── churn_library.py`\
`├── churn_script_logging_and_test.py`\
`├── constants.py`\
`└── requirements_py3.8.txt`\

The 4 main files are:
- churn_library.py
  - It is the main library containing all the functions to perform EDA plots, feature engineering, model training (for both models Random Forest and Logistic Regression) and model evaluation plots using ROC curves, feature importance analysis, Classification Report and the SHAP library for model explainability purposes.
- churn_script_logging_and_test.py
  - Module containing functions to logging and testing steps, inputs and outputs of each one of the churn_library functions.
- constants.py
  - File with constants and parameters needed to run and test the whole system.
- requirements_py3.8.txt: Python dependencies for version 3.8

The 4 folders and its contents are:
- data: contains the csv file used to train the model
- images:
  - eda: contains eda plots for the main features (distributions plot and heatmap correlation plot) 
  - results: contains feature importance analysis, classification reports, ROC curves and SHAP explainability results.
- logs: log file as result of the implemented logger.
- models: both Random Forest and Logistic Regression models saved into Pickle format.

## Running Files
How do you run your files? What should happen when you run your files?

In order to build both models, results and plots, do the following:
- Install dependencies from the file requirements_py3.8.txt
- Run 'python churn_library.py' in your terminal

In order to test the churn_library.py, run:
- 'pytest churn_script_logging_and_tests.py`



