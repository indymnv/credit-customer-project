# library doc string


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
            category_var_lst.append(cat_var_groups.loc[val])

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
    pass


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
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
