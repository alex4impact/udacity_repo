import os 
import sys

# Parent Folder 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

dir = os.path.dirname(os.path.realpath(__file__))

CSV_FILE = "bank_data.csv"
DF_PATH = f'{dir}/data/{CSV_FILE}'

PLOTS = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct','heatmap']

PLOTS_PATH = f'{dir}/images/eda/'

MODELS_PATH = f'{dir}/models/'

RESULTS_PATH = f'{dir}/images/results/'

MODELS = ['rfc_model.pkl','logistic_model.pkl']

CATEGORY_LIST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]

RESPONSE = 'Churn'

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        f'Gender_{RESPONSE}', f'Education_Level_{RESPONSE}',
        f'Marital_Status_{RESPONSE}', f'Income_Category_{RESPONSE}',
        f'Card_Category_{RESPONSE}']

PARAM_GRID = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

