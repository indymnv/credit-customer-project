"""
This file provide a end to end machine learning workflow with a classification model for a customer churn prediction
name: Indy Navarro
date: None
"""

import os
import logging

import config
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		cls.perform_eda(df)

		assert os.path.isfile("./images/churn_hist.png")
		assert os.path.isfile("./images/Customer_age_hist.png")
		assert os.path.isfile("./images/marital_status_bar.png")
		assert os.path.isfile("./images/Total_trans_distplot.png")
		assert os.path.isfile("./images/df_corr_heatmap.png")


	except FileNotFoundError as err:
		logging.error("Testing test_eda: The file wasn't found")
		raise err



def test_encoder_helper():
	'''
	test encoder helper
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		df = cls.encoder_helper(df, config.CATEGORY_LIST, response=None)
		assert df.shape[0]== 10127
		assert df.shape[1]== 28

	except AssertionError as err:
		logging.error("Testing encoder_helper: The file doesn't appear to have the exact number of rows and columns")
		raise err


def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	
	try:
		df = cls.import_data("./data/bank_data.csv")
		df = cls.encoder_helper(df, config.CATEGORY_LIST, response=None)
		X_train, X_test, y_train, y_test  = cls.perform_feature_engineering(df, response = None)
		assert X_train.shape[0]== 7088
		assert X_train.shape[1]== 19
		assert X_test.shape[0]== 3039
		assert X_test.shape[1]== 19
		assert y_train.shape[0]== 7088
		assert y_test.shape[0]== 3039

	except AssertionError as err:

		logging.error("Testing perform_feature_engineering: The split doesn't appear to have the exact number of rows and columns")
		raise err




def test_train_models():
	'''
	test train_models
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		df = cls.encoder_helper(df, config.CATEGORY_LIST, response=None)
		X_train, X_test, y_train, y_test  = cls.perform_feature_engineering(df, response = None)
		y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = cls.train_models(X_train, X_test, y_train, y_test)

		assert y_train_preds_rf.shape[0] > 0
		assert y_test_preds_rf.shape[0] > 0
		assert y_train_preds_lr.shape[0] > 0
		assert y_test_preds_lr.shape[0] > 0

	except AssertionError as err:

		logging.error("Testing perform_feature_engineering: The split doesn't appear to have rows")
		raise err
	
	try:

		assert os.path.isfile("./models/logistic_model.pkl")
		assert os.path.isfile("./models/rfc_model.pkl")
	
	except FileNotFoundError as err:
		logging.error("Testing test_eda: The model wasn't found")
		raise err


if __name__ == "__main__":
	test_import(cls.import_data)
	#test_eda()
	#test_encoder_helper








