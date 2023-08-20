from submit import my_predict
import numpy as np
import pandas as pd
import time as tm
import os

%load_ext autoreload
%autoreload 2

df_test = pd.read_csv( "dummy_test.csv" )

n_trials = 5
t_test = 0

for t in range( n_trials ):
  tic = tm.perf_counter()
  df_feat = df_test.drop( [ "OZONE", "NO2" ], axis = "columns" )
  ( pred_o3, pred_no2 ) = my_predict( df_feat )
  toc = tm.perf_counter()
  t_test += toc - tic

t_test /= n_trials

gold_o3 = df_test[ "OZONE" ].to_numpy()
gold_no2 = df_test[ "NO2" ].to_numpy()

mae_o3 = np.mean( np.abs( pred_o3 - gold_o3 ) )
mae_no2 = np.mean( np.abs( pred_no2 - gold_no2 ) )

print( t_test, mae_o3, mae_no2 )
