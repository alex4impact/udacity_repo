"""
this library contains all the functions used for processing, feature engineering, training and evaluation of the predict customer churn model

author: Alex Carvalho
date: September 2022
"""

# import libraries
import os
from constants import DF_PATH, PLOTS, PLOTS_PATH, CATEGORY_LIST, RESPONSE, KEEP_COLS

os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    Args:
            pth: a path to the csv
    Returns:
            df: pandas dataframe
    '''	
    
    import pandas as pd

    df = pd.read_csv(pth, index_col=0)

    df[RESPONSE] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    return df


def perform_eda(df, plots, plots_path):
    '''
    perform eda on df and save figures to images folder
    
    Args:
            df: pandas dataframe
    Returns:
            None
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    
    
    for plot in plots:
        plt.figure(figsize=(20,10))
        plot_filename_path = f'{plots_path + str.lower(plot)}_distribution.png'
        
        if plot == 'Marital_Status':
            df[plot].value_counts('normalize').plot(kind='bar')
            plt.savefig(plot_filename_path)
            plt.close()
        elif plot == 'Total_Trans_Ct':
            sns.histplot(df[plot], stat='density', kde=True)
            plt.savefig(plot_filename_path)
            plt.close()
        elif plot == 'heatmap':
            sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
            plt.savefig(f'{plots_path + str.lower(plot)}.png')
            plt.close()
        else:
            df[plot].hist()
            plt.savefig(plot_filename_path)
            plt.close()
    return None


def encoder_helper(df, category_list, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
            df: pandas dataframe
            CATEGORY_LIST: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
            df: pandas dataframe with new columns for
    '''
    
    import numpy as np
    
    for cat in category_list:
        new_cat_name = f'{cat}_{response}'
        df[new_cat_name] = df.groupby(cat)[response].transform(np.mean)
        
    return df


def perform_feature_engineering(df, category_list, keep_cols, response):
    '''
    Args:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = encoder_helper(df, category_list, response)
    
    y = df[response]
    X = pd.DataFrame()
    
    X[keep_cols] = df[keep_cols]
    
    # return train test split 
    return train_test_split(X, y, test_size= 0.3, random_state=42)
    

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    
    Args:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    Returns:
        None
    '''
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    
    filename_rfc = './mlops_project_1_production_ready_code_for_ml/images/results/rf_results.png'
    filename_lrc = './mlops_project_1_production_ready_code_for_ml/images/results/logistic_results.png'
    
    for model in [('Logistic Regression ', y_test_preds_lr, y_train_preds_lr, filename_lrc),
                ('Random Forest ', y_test_preds_rf, y_train_preds_rf, filename_rfc)]:
        plt.rc('figure', figsize=(8, 8))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str(f'{model[0]}Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, model[1])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model[0]}Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, model[2])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(model[3], bbox_inches = "tight")
        plt.close()
    
    return None


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    
    Args:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    Returns:
             None
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    # Save plot
    plt.savefig(output_pth)
    plt.close()
    
    return None


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    
    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    Returns:
        None
    '''
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import plot_roc_curve
    import matplotlib.pyplot as plt
    import shap
    import joblib
    
    ## Train models
    # random forest with grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    
    ## train cv_rfc
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    ## train lrc
    lrc.fit(X_train, y_train)
    
    ## Store cv_rfc model
    cv_rfc_filename = './mlops_project_1_production_ready_code_for_ml/models/rfc_model.pkl'
    with open(cv_rfc_filename, 'wb') as file_cv_rfc:  
        joblib.dump(cv_rfc.best_estimator_, file_cv_rfc)
        
    ## Store lrc model
    lrc_filename = './mlops_project_1_production_ready_code_for_ml/models/logistic_model.pkl'
    with open(lrc_filename, 'wb') as file_lrc:  
        joblib.dump(lrc, file_lrc)
    
    ## Generate cv_rfc model predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    ## Generate lrc model predictions
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    ## Generate classification report images
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    ## Generate feature importance image
    output_pth = './mlops_project_1_production_ready_code_for_ml/images/results/feature_importances.png'
    feature_importance_plot(cv_rfc.best_estimator_, X_train, output_pth)
    
    ## Generate ROC curves images
    # load saved models
    rfc_model = joblib.load('./mlops_project_1_production_ready_code_for_ml/models/rfc_model.pkl')
    lr_model = joblib.load('./mlops_project_1_production_ready_code_for_ml/models/logistic_model.pkl')
    
    # Generate ROC plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.close()
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    # Save plot
    plt.savefig('./mlops_project_1_production_ready_code_for_ml/images/results/roc_curve_result.png')
    plt.close()
    
    ## Generate SHAP plot
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.gcf()
    plt.savefig('./mlops_project_1_production_ready_code_for_ml/images/results/shap_explainer_result.png', bbox_inches = "tight")
    plt.close()

    return None

if __name__ == "__main__":
    # import data
    df = import_data(DF_PATH)
    
    # perform eda
    perform_eda(df, PLOTS, PLOTS_PATH)
    
    # perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, CATEGORY_LIST, KEEP_COLS, RESPONSE)
    # train model
    train_models(X_train, X_test, y_train, y_test)