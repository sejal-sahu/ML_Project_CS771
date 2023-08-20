import pandas as pd
import numpy as np
from sklearn import metrics

from joblib import Parallel, delayed
import joblib

def my_predict(test_data):
    x_test = test_data[[ "temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2" ]]
    x_testf = test_data[[ "Time", "temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2" ]]

    # split time into date, hours and minutes
    x_testf["Time"] = pd.to_datetime(x_testf["Time"])
    x_testf["date"] = x_testf["Time"].dt.date
    x_testf["hour"] = x_testf["Time"].dt.hour
    x_testf["minute"] = x_testf["Time"].dt.minute
    x_testf = x_testf.drop(columns=["Time"])

    # keep only the first part of the date
    x_testf["date"] = x_testf["date"].astype(str)
    x_testf["date"] = x_testf["date"].str.split("-").str[1]
    
    x_testf["date"] = x_testf["date"].astype(int)

    model_NO2 = joblib.load("model_NO2.pkl")
    model_O3 = joblib.load("model_O3.pkl")

    pred_o3 = model_O3.predict(x_testf)
    pred_no2 = model_NO2.predict(x_testf)
    return ( pred_o3, pred_no2 )
