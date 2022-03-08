# library doc string
"""
This file provide a end to end machine learning workflow with a classification model for a customer churn prediction
name: Indy Navarro
date: 
"""

# import libraries
#import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import config

def import_data(pth = config.DATA_PATH):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    ''' 
   
    
    # Get Churn Histogram
    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig(r"./images/churn_hist.png")

    # Get Customer Age Histogram
    plt.figure(figsize=(20,10))
    df['Customer_Age'].hist()
    plt.savefig(r"./images/Customer_age_hist.png")

    # Get marital status bar
    plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(r"./images/marital_status_bar.png")

    #Get Total Trans distplot
    plt.figure(figsize=(20,10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(r"./images/Total_trans_distplot.png")

    # Get heatmap DataFrame
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(r"./images/df_corr_heatmap.png")



def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category_var in category_lst:

        cat_var_lst = []
        cat_var_groups = df.groupby(category_var).mean()['Churn']

        for val in df[category_var]:
            cat_var_lst.append(cat_var_groups.loc[val])

        df[category_var+'_Churn'] = cat_var_lst
    
    return df

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = df["churn"]
    X = pd.DataFrame()
    X[config.KEEP_COLS] = df[config.KEEP_COLS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test 

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r"./images/report_rf.png")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r"./images/report_lr.png")




def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    #rfc_model = joblib.load('./models/rfc_model.pkl') #Then replace for model

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
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

    plt.savefig(r"./images/feature_importance.png")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
              
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()


    cv_rfc = GridSearchCV(estimator=rfc, param_grid= config.PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Plot roc-auc curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig(r"./images/ROC_lr_test.png")

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(r"./images/ROC_lr_and_rf_test.png")

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr

if __name__ == "__main__":

    df = import_data(config.DATA_PATH)

    perform_eda(df)

    df = encoder_helper(df, category_lst = config.CATEGORY_LIST)

    X_train, X_test, y_train, y_test  = perform_feature_engineering(df)

    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(X_train, X_test, y_train, y_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    rfc_model = joblib.load('./models/rfc_model.pkl')

    feature_importance_plot(rfc_model, X_train, output_pth = config.IMAGE_PATH)







