import os 
import sys

# Parent Folder 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

dir = os.path.dirname(os.path.realpath(__file__))

CSV_FILE = "bank_data.csv"
DF_PATH = f'{dir + "/data/" + CSV_FILE}'

PLOTS = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct','heatmap']

PLOTS_PATH = f'{dir + "/images/eda/"}'
