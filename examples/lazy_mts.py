import os 
import mlsauce as ms 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
try: 
    from statsmodels.tsa.base.datetools import dates_from_str
except ImportError:
    ModuleNotFoundError

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# some example data
mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
print(mdata.head())
mdata = mdata[['realgovt', 'tbilrate', 'cpi']]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

n = data.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]


regr_mts = ms.LazyBoostingMTS(verbose=0, ignore_warnings=True, 
                      lags = 20, n_hidden_features=7, n_clusters=2,
                      type_pi="scp2-block-bootstrap", 
                      #kernel="gaussian",
                      replications=250, 
                      show_progress=False, preprocess=False, 
                      sort_by="WINKLERSCORE",)
models = regr_mts.fit(df_train, df_test)

print(models[["RMSE", "WINKLERSCORE", "Time Taken"]])