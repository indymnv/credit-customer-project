"""
This file provide several test for respective functions in churn_library.py 
author: Indy Navarro
date: 26 feb 2022
"""

import os
import logging

import warnings
warnings.filterwarnings("ignore")

import config
import churn_library as cls





logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_data.shape[0] > 0
        assert bank_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(bank_data)

        assert os.path.isfile("./images/churn_hist.png")
        assert os.path.isfile("./images/Customer_age_hist.png")
        assert os.path.isfile("./images/marital_status_bar.png")
        assert os.path.isfile("./images/Total_trans_distplot.png")
        assert os.path.isfile("./images/data_corr_heatmap.png")
        logging.info("Testing eda files: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing test_eda: The file wasn't found")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        bank_data = cls.encoder_helper(bank_data, config.CATEGORY_LIST, response=None)
        assert bank_data.shape[0] == 10127
        assert bank_data.shape[1] == 28
        logging.info("Testing encoder helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have the exact number of rows and columns")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        bank_data = cls.encoder_helper(bank_data, config.CATEGORY_LIST, response=None)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            bank_data, response=None)
        assert X_train.shape[0] == 7088
        assert X_train.shape[1] == 19
        assert X_test.shape[0] == 3039
        assert X_test.shape[1] == 19
        assert y_train.shape[0] == 7088
        assert y_test.shape[0] == 3039
        logging.info("Testing feature engineering: SUCCESS")

    except AssertionError as err:

        logging.error(
            "Testing perform_feature_engineering: The split doesn't appear to have the exact number of rows and columns")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        bank_data = cls.encoder_helper(bank_data, config.CATEGORY_LIST, response=None)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            bank_data, response=None)
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = cls.train_models(
            X_train, X_test, y_train, y_test)

        assert y_train_preds_rf.shape[0] > 0
        assert y_test_preds_rf.shape[0] > 0
        assert y_train_preds_lr.shape[0] > 0
        assert y_test_preds_lr.shape[0] > 0
        logging.info("Testing train:models: SUCCESS")

    except AssertionError as err:

        logging.error(
            "Testing perform_feature_engineering: The split doesn't appear to have rows")
        raise err

    try:

        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")

    except FileNotFoundError as err:
        logging.error("Testing test_eda: The model wasn't found")
        raise err


if __name__ == "__main__":

    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
