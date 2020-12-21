#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@editor: Wei Wei
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: main.py
@time: 2020/11/29 18:20
@desc: main function for running savedModel
'''
from cleaner import BuildDataset
from models import Model, XGBoost
from models import FineTuning
from util import saveResult, trainvalLossPlot, getBasePath
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
import joblib
import argparse


def savemodel(model, modelname):
    joblib.dump(model, '%s/savedModel/%s.joblib' % (getBasePath(), modelname), compress=0)


def loadModel(model_name):
    return joblib.load('%s/savedModel/%s.joblib' % (getBasePath(), model_name))


def finetune(model, X, y):
    # grid search
    param_grid = {'min_child_weight': [6, 7], 'subsample': [0.5], 'gamma': [0.05], 'max_depth': [6, 7],
                  'learning_rate': [0.01, 0.1], 'n_estimators': [500, 600]}
    fitter = FineTuning(model, cv=10, param_grid=param_grid)
    model_current, params, best_score = fitter.finetuning(X, y)
    print('best_params')
    print(params)
    print('best_score: %.3f' % best_score)
    savemodel(model_current, 'xgboost-current')
    return model_current, params


def predict(testId, testX, model, stander):
    '''
    Do prediction
    :param testId:
    :param testX:
    :param model: The chosen best model
    :return: None
    '''
    ypred = model.predict(testX)
    ypred = stander.inverse_log10_y(ypred)

    # save prediction to csv
    saveResult(testId, ypred)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',default='xgboost-current', type=str, help='model name, like xgboost-1, pick one name from savedModel')
    parser.add_argument('-t', '--task', default='train', type=str, help="set task, train or reproduce. If task==train, model will be saved as xgboost-current")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ########## For building features
    '''
    This region is for choosing the best way to build features. After running those
    below, you will get training scores and test scores. Do test it use different
    paramsDict and . To get details of what 'paramsDict' support, check the annotations 
    in trainVal().
    
    Example:
    >> buildDict = {'kbest':-1,
                       'featuresEng':'num+onehot',
                       'dropOutlierRatio':0.25,
                       'discreteMethod': None,
                       'bins':200}
    >> trainVal(buildDict=buildDict)  
    '''

    ########## For model params
    '''
    This region is for fine tuning model, after choosing the method of building features,
    use finetune(model,X,y)
    
    Example:
    
    >> model_best, params = finetune(model,X,y)
    '''
    ########## for prediction
    '''
    The model_best getting from finetune(model,X,y) will be used for prediction
    
    Example:
    >> predict(testId,testX,model_best,stander)
    '''
    opt = parse_command_line()
    task = opt.task

    # ------------------- load data with best features building rules.
    builder = BuildDataset(kbest=35, featuresEng='num+onehot',
                           dropOutlierRatio=0.39, discreteMethod=None, imgFeaDim=0)
    X, Y, Ylog, testX, testId, stander = builder.getData()

    if task=='train':
        # ----------------------fine tune
        model = XGBRegressor()
        model_current = finetune(model, X, Ylog)[0]
        # # ----------------------reproduce best result on kaggle competition
        # model_best = loadModel('xgboost-current') #
        # model_best.fit(X, Ylog)
        # predict(testId, testX, model_best, stander)
    elif task=='reproduce':
        model_best = loadModel(opt.model_name)  # xgboost-1, xgboost-current
        # model_best.fit(X, Ylog)
        predict(testId, testX, model_best, stander)