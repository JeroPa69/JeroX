import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from scipy.stats import norm
from tabulate import tabulate
from sklearnex import patch_sklearn
from time import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
patch_sklearn()
np.warnings.filterwarnings('ignore')
plt.style.use('dark_background')

##
# Preprocessing

sheets = ['FX', 'EQC', 'G', 'Vol', 'Ldn Open', 'Bid', 'Ask', 'Px', 'H', 'L', 'IR', 'EQV', 'EQM',
          'EQO', 'ToT', 'S3M', 'S6M', 'VOL1M', 'VOL3M', 'VOL1Y', '10y', 'FI', 'CPI2', 'EQO', 'EQH',
          '5y', '2y', '1y', 'EY', 'DCAR', 'LCAR', 'COM', 'CB', 'INF', 'DY', 'CPI', 'ReeR', 'EQL',
          'PxB', 'RR3M', 'BF3M', 'PE', 'F3M', 'NY Open', 'Asia Close', 'F3MH', 'F3ML', 'F3MO', 'EQS',
          'USD', 'EQMC', 'EQBV', 'EQCAP']

data = pd.read_excel(r'/Users/marianoarrieta/Desktop/ModelValues1.xlsx', sheets, index_col=0)


def data_organizer(df, sheet):
    name = df[sheet].copy()
    name = name.drop(name.columns[0], axis=1)
    name = name.drop(name.index[[0, 1, 2]])
    name = name.set_index(name.columns[0]).rename_axis('Date')
    return name


FX = data_organizer(data, 'FX').iloc[:, : 22]
FX['RUB']['2022-02-24'::] = np.nan
Asia_Close = data_organizer(data, 'Asia Close').iloc[:, : 22]
Asia_Close['RUB']['2022-02-24'::] = np.nan
NY_Open = data_organizer(data, 'NY Open').iloc[:, : 22]
NY_Open['RUB']['2022-02-24'::] = np.nan

Ask_FX = data_organizer(data, 'Ask').iloc[:, : 22]
Ask_FX['RUB']['2022-02-24'::] = np.nan
Bid_FX = data_organizer(data, 'Bid').iloc[:, : 22]
Bid_FX['RUB']['2022-02-24'::] = np.nan

F3M = data_organizer(data, 'F3M').iloc[:, : 22]
F3M['RUB']['2022-02-24'::] = np.nan
F3M['AUD'] = 1 / F3M['AUD']
F3M['NZD'] = 1 / F3M['NZD']
F3M['EUR'] = 1 / F3M['EUR']
F3M['GBP'] = 1 / F3M['GBP']

F3MO = data_organizer(data, 'F3MO').iloc[:, : 22]
F3MO['RUB']['2022-02-24'::] = np.nan
F3MO['AUD'] = 1 / F3MO['AUD']
F3MO['NZD'] = 1 / F3MO['NZD']
F3MO['EUR'] = 1 / F3MO['EUR']
F3MO['GBP'] = 1 / F3MO['GBP']

F3ML = data_organizer(data, 'F3ML').iloc[:, : 22]
F3ML['RUB']['2022-02-24'::] = np.nan
F3ML['AUD'] = 1 / F3ML['AUD']
F3ML['NZD'] = 1 / F3ML['NZD']
F3ML['EUR'] = 1 / F3ML['EUR']
F3ML['GBP'] = 1 / F3ML['GBP']

F3MH = data_organizer(data, 'F3MH').iloc[:, : 22]
F3MH['RUB']['2022-02-24'::] = np.nan
F3MH['AUD'] = 1 / F3MH['AUD']
F3MH['NZD'] = 1 / F3MH['NZD']
F3MH['EUR'] = 1 / F3MH['EUR']
F3MH['GBP'] = 1 / F3MH['GBP']

USD = data_organizer(data, 'USD').iloc[:, : 10]
REER = data_organizer(data, 'ReeR').iloc[:, : 22]
PE = data_organizer(data, 'PE').iloc[:, : 22]
PER = PE.div(data_organizer(data, 'PE')['USD'], axis=0)
TR = data_organizer(data, 'Px').iloc[:, : 22]
TR['RUB']['2022-02-24'::] = np.nan
FX_Open = data_organizer(data, 'Ldn Open').iloc[:, : 22]
FX_Open['RUB']['2022-02-24'::] = np.nan
EQC = data_organizer(data, 'EQC').iloc[:, : 22]
EQC['RUB']['2022-02-24'::] = np.nan
EQCR = EQC.div(data_organizer(data, 'EQC')['USD'], axis=0)
EQCR['RUB']['2022-02-24'::] = np.nan
EQO = data_organizer(data, 'EQO').iloc[:, : 22]
EQO['RUB']['2022-02-24'::] = np.nan
EQH = data_organizer(data, 'EQH').iloc[:, : 22]
EQH['RUB']['2022-02-24'::] = np.nan
EQL = data_organizer(data, 'EQL').iloc[:, : 22]
EQL['RUB']['2022-02-24'::] = np.nan
EQV = data_organizer(data, 'EQV').iloc[:, : 22]
EQV['RUB']['2022-02-24'::] = np.nan
EQMC = data_organizer(data, 'EQMC').iloc[:, : 22]
EQMC['RUB']['2022-02-24'::] = np.nan
COM = data_organizer(data, 'COM').iloc[:, : 22]

EQM = data_organizer(data, 'EQM').iloc[:, : 22]
EQM['RUB']['2022-02-24'::] = np.nan
EQMR = EQM.div(data_organizer(data, 'EQM')['USD'], axis=0)
EQMR['RUB']['2022-02-24'::] = np.nan

EQS = data_organizer(data, 'EQS').iloc[:, : 22]
EQS['RUB']['2022-02-24'::] = np.nan
EQBV = data_organizer(data, 'EQBV').iloc[:, : 22]
EQBV['RUB']['2022-02-24'::] = np.nan
EQCAP = data_organizer(data, 'EQCAP').iloc[:, : 22]
EQCAP['RUB']['2022-02-24'::] = np.nan

VOL = data_organizer(data, 'Vol').iloc[:, : 22]
VOL = VOL.fillna(VOL.mean())
VOL['RUB']['2022-02-24'::] = np.nan
IR = data_organizer(data, 'IR').iloc[:, : 22]
IR['RUB']['2022-02-24'::] = np.nan
IRR = IR.div(data_organizer(data, 'IR')['USD'], axis=0)
IRR['RUB']['2022-02-24'::] = np.nan

DCAR = data_organizer(data, 'DCAR').iloc[:, : 22]
DCAR['RUB']['2022-02-24'::] = np.nan
LCAR = data_organizer(data, 'LCAR').iloc[:, : 22]
LCAR['RUB']['2022-02-24'::] = np.nan
VOL1M = data_organizer(data, 'VOL1M').iloc[:, : 22]
VOL1M['RUB']['2022-02-24'::] = np.nan
VOL3M = data_organizer(data, 'VOL3M').iloc[:, : 22]
VOL3M['RUB']['2022-02-24'::] = np.nan
VOL1Y = data_organizer(data, 'VOL1Y').iloc[:, : 22]
VOL1Y['RUB']['2022-02-24'::] = np.nan

Y10 = data_organizer(data, '10y').iloc[:, : 22]
Y10['RUB']['2022-02-24'::] = np.nan
Y10R = Y10.div(data_organizer(data, '10y')['USD'], axis=0)
Y10R['RUB']['2022-02-24'::] = np.nan
Y5 = data_organizer(data, '5y').iloc[:, : 22]
Y5['RUB']['2022-02-24'::] = np.nan
Y5R = Y5.div(data_organizer(data, '5y')['USD'], axis=0)
Y5R['RUB']['2022-02-24'::] = np.nan
Y2 = data_organizer(data, '2y').iloc[:, : 22]
Y2['RUB']['2022-02-24'::] = np.nan
Y2R = Y2.div(data_organizer(data, '2y')['USD'], axis=0)
Y2R['RUB']['2022-02-24'::] = np.nan
Y1 = data_organizer(data, '1y').iloc[:, : 22]
Y1['RUB']['2022-02-24'::] = np.nan
Y1R = Y1.div(data_organizer(data, '1y')['USD'], axis=0)
Y1R['RUB']['2022-02-24'::] = np.nan
EY = data_organizer(data, 'EY').iloc[:, : 22]
EY['RUB']['2022-02-24'::] = np.nan
EYR = EY.div(data_organizer(data, 'EY')['USD'], axis=0)
EYR['RUB']['2022-02-24'::] = np.nan

Ext_Factors = data_organizer(data, 'G')
H = data_organizer(data, 'H')
H['RUB']['2022-02-24'::] = np.nan
L = data_organizer(data, 'L')
L['RUB']['2022-02-24'::] = np.nan
BID = data_organizer(data, 'Bid').iloc[:, : 22]
BID['RUB']['2022-02-24'::] = np.nan
ASK = data_organizer(data, 'Ask').iloc[:, : 22]
ASK['RUB']['2022-02-24'::] = np.nan

CB = data_organizer(data, 'CB').iloc[:, : 22]
CB['RUB']['2022-02-24'::] = np.nan
INF = data_organizer(data, 'INF').iloc[:, : 22]
INF['RUB']['2022-02-24'::] = np.nan

TOT = data_organizer(data, 'ToT').iloc[:, : 22]
TOT['RUB']['2022-02-24'::] = np.nan
TOTR = TOT.subtract(data_organizer(data, 'ToT')['USD'], axis=0)
TOTR['RUB']['2022-02-24'::] = np.nan
S3M = data_organizer(data, 'S3M').iloc[:, : 22]
S3M['RUB']['2022-02-24'::] = np.nan
S3MR = S3M.subtract(data_organizer(data, 'S3M')['USD'], axis=0)
S3MR['RUB']['2022-02-24'::] = np.nan
S6M = data_organizer(data, 'S6M').iloc[:, : 22]
S6M['RUB']['2022-02-24'::] = np.nan
S6MR = S6M.subtract(data_organizer(data, 'S6M')['USD'], axis=0)
S6MR['RUB']['2022-02-24'::] = np.nan
DY = data_organizer(data, 'DY').iloc[:, : 22]
DY['RUB']['2022-02-24'::] = np.nan
DYR = DY.subtract(data_organizer(data, 'DY')['USD'], axis=0)
DYR['RUB']['2022-02-24'::] = np.nan

CPI = data_organizer(data, 'CPI').iloc[:, : 22]
CPI['RUB']['2022-02-24'::] = np.nan
CPIR = CPI.subtract(data_organizer(data, 'CPI')['USD'], axis=0)
CPIR['RUB']['2022-02-24'::] = np.nan

CPI2 = data_organizer(data, 'CPI2').iloc[:, : 22]
CPI2['RUB']['2022-02-24'::] = np.nan
CPI2R = CPI.div(data_organizer(data, 'CPI2')['USD'], axis=0)
CPI2R['RUB']['2022-02-24'::] = np.nan

PB = data_organizer(data, 'PxB').iloc[:, : 22]
PB['RUB']['2022-02-24'::] = np.nan
PBR = PB.subtract(data_organizer(data, 'PxB')['USD'], axis=0)
PBR['RUB']['2022-02-24'::] = np.nan

RR3M = data_organizer(data, 'RR3M').iloc[:, : 22]
RR3M['RUB']['2022-02-24'::] = np.nan
BF3M = data_organizer(data, 'BF3M').iloc[:, : 22]
BF3M['RUB']['2022-02-24'::] = np.nan

FI = data_organizer(data, 'FI').iloc[:, : 22]
FI['RUB']['2022-02-24'::] = np.nan
FIR = FI.div(data_organizer(data, 'FI')['USD'], axis=0)
FIR['RUB']['2022-02-24'::] = np.nan

FX['MXN'].index = pd.to_datetime(FX.index)
FX['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
FX['TRY'].index = pd.to_datetime(FX.index)
FX['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
FX['CNH'].index = pd.to_datetime(FX.index)
FX['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
FX['RUB'].index = pd.to_datetime(FX.index)
FX['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
FX['THB'].index = pd.to_datetime(FX.index)
FX['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
FX['INR'].index = pd.to_datetime(FX.index)
FX['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
FX['RON'].index = pd.to_datetime(FX.index)
FX['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
FX['ILS'].index = pd.to_datetime(FX.index)
FX['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

F3M['MXN'].index = pd.to_datetime(F3M.index)
F3M['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
F3M['TRY'].index = pd.to_datetime(F3M.index)
F3M['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
F3M['CNH'].index = pd.to_datetime(F3M.index)
F3M['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
F3M['RUB'].index = pd.to_datetime(F3M.index)
F3M['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
F3M['THB'].index = pd.to_datetime(F3M.index)
F3M['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
F3M['INR'].index = pd.to_datetime(F3M.index)
F3M['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
F3M['RON'].index = pd.to_datetime(F3M.index)
F3M['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
F3M['ILS'].index = pd.to_datetime(F3M.index)
F3M['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

F3MH['MXN'].index = pd.to_datetime(F3MH.index)
F3MH['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
F3MH['TRY'].index = pd.to_datetime(F3MH.index)
F3MH['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
F3MH['CNH'].index = pd.to_datetime(F3MH.index)
F3MH['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
F3MH['RUB'].index = pd.to_datetime(F3MH.index)
F3MH['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
F3MH['THB'].index = pd.to_datetime(F3MH.index)
F3MH['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
F3MH['INR'].index = pd.to_datetime(F3MH.index)
F3MH['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
F3MH['RON'].index = pd.to_datetime(F3MH.index)
F3MH['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
F3MH['ILS'].index = pd.to_datetime(F3MH.index)
F3MH['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

F3ML['MXN'].index = pd.to_datetime(F3ML.index)
F3ML['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
F3ML['TRY'].index = pd.to_datetime(F3ML.index)
F3ML['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
F3ML['CNH'].index = pd.to_datetime(F3ML.index)
F3ML['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
F3ML['RUB'].index = pd.to_datetime(F3ML.index)
F3ML['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
F3ML['THB'].index = pd.to_datetime(F3ML.index)
F3ML['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
F3ML['INR'].index = pd.to_datetime(F3ML.index)
F3ML['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
F3ML['RON'].index = pd.to_datetime(F3ML.index)
F3ML['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
F3ML['ILS'].index = pd.to_datetime(F3ML.index)
F3ML['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

F3MO['MXN'].index = pd.to_datetime(F3MO.index)
F3MO['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
F3MO['TRY'].index = pd.to_datetime(F3MO.index)
F3MO['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
F3MO['CNH'].index = pd.to_datetime(F3MO.index)
F3MO['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
F3MO['RUB'].index = pd.to_datetime(F3MO.index)
F3MO['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
F3MO['THB'].index = pd.to_datetime(F3MO.index)
F3MO['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
F3MO['INR'].index = pd.to_datetime(F3MO.index)
F3MO['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
F3MO['RON'].index = pd.to_datetime(F3MO.index)
F3MO['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
F3MO['ILS'].index = pd.to_datetime(F3MO.index)
F3MO['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

FX_Open['MXN'].index = pd.to_datetime(FX_Open.index)
FX_Open['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
FX_Open['TRY'].index = pd.to_datetime(FX_Open.index)
FX_Open['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
FX_Open['CNH'].index = pd.to_datetime(FX_Open.index)
FX_Open['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
FX_Open['RUB'].index = pd.to_datetime(FX_Open.index)
FX_Open['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
FX_Open['THB'].index = pd.to_datetime(FX_Open.index)
FX_Open['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
FX_Open['INR'].index = pd.to_datetime(FX_Open.index)
FX_Open['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
FX_Open['RON'].index = pd.to_datetime(FX_Open.index)
FX_Open['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
FX_Open['ILS'].index = pd.to_datetime(FX.index)
FX_Open['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

Asia_Close['MXN'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
Asia_Close['TRY'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
Asia_Close['CNH'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
Asia_Close['RUB'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
Asia_Close['THB'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
Asia_Close['INR'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['INR'].loc['1/1/1993':'1/1/2010'] = np.nan

Asia_Close['MXN'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
Asia_Close['TRY'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
Asia_Close['CNH'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
Asia_Close['RUB'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
Asia_Close['THB'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
Asia_Close['INR'].index = pd.to_datetime(Asia_Close.index)
Asia_Close['INR'].loc['1/1/1993':'1/1/2010'] = np.nan

NY_Open['MXN'].index = pd.to_datetime(NY_Open.index)
NY_Open['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
NY_Open['TRY'].index = pd.to_datetime(NY_Open.index)
NY_Open['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
NY_Open['CNH'].index = pd.to_datetime(NY_Open.index)
NY_Open['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
NY_Open['RUB'].index = pd.to_datetime(NY_Open.index)
NY_Open['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
NY_Open['THB'].index = pd.to_datetime(NY_Open.index)
NY_Open['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
NY_Open['INR'].index = pd.to_datetime(NY_Open.index)
NY_Open['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
NY_Open['RON'].index = pd.to_datetime(NY_Open.index)
NY_Open['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
NY_Open['ILS'].index = pd.to_datetime(NY_Open.index)
NY_Open['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

TR['MXN'].index = pd.to_datetime(TR.index)
TR['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
TR['TRY'].index = pd.to_datetime(TR.index)
TR['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
TR['CNH'].index = pd.to_datetime(TR.index)
TR['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
TR['RUB'].index = pd.to_datetime(TR.index)
TR['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
TR['THB'].index = pd.to_datetime(TR.index)
TR['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
TR['INR'].index = pd.to_datetime(TR.index)
TR['INR'].loc['1/1/1993':'1/1/2010'] = np.nan
TR['RON'].index = pd.to_datetime(TR.index)
TR['RON'].loc['1/1/1993':'1/1/2006'] = np.nan
TR['ILS'].index = pd.to_datetime(FX.index)
TR['ILS'].loc['1/1/1993':'1/1/2004'] = np.nan

# Transaction costs

tc = [0.0004, 0.0001, 0.0002, 0.0003, 0.0007, 0.0003, 0.0002, 0.0002,
      0.00025, 0.0005, 0.00025, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
      0.0001, 0.0001, 0.0001, 0.0001, 0.0004, 0.0002]

# Based on JP Morgan transaction costs/market impact model

'adv = average daily volume at trading'
'spd = average transaction cost spread'
'spd_frac = '

def TransCosts(position=1, adv=0.1, day_frac=1.0, spd=tc,
               spd_frac=1, ann_vol=VOL / 100, omega=0.92,
               alpha=350, beta=0.370, gamma=1.05):
    PoV = (position / (adv * day_frac)) / 100
    I = alpha * (PoV ** beta) * (ann_vol ** gamma) / 100000
    MI = I * omega * (2 * PoV) / (1 + PoV) + (1 - omega) * I + (np.array(spd) * spd_frac)
    return MI

slippage = TransCosts()

def getweights(signals):
    signals[signals > 0] = signals[signals > 0].div(signals[signals > 0].sum(axis=1), axis=0)
    signals[signals < 0] = -(signals[signals < 0].div(signals[signals < 0].sum(axis=1), axis=0))
    return signals.fillna(0)

def voladjew(signals):
    long_signal = signals[signals > 0] * (10 / VOL)
    long = long_signal.div(long_signal.sum(axis=1), axis=0)
    short_signal = signals[signals < 0] * (10 / VOL)
    short = short_signal.div(short_signal.sum(axis=1), axis=0)
    return (long.fillna(0) - short.fillna(0))

def rank_w(Px):
    rnk = Px.rank(ascending=True, axis=1)
    rnk_w = rnk.subtract(rnk.mean(axis=1), axis=0)
    scalar = rnk_w[rnk_w>0].sum(axis=1)
    rnk_l = rnk_w[rnk_w > 0].div(scalar, axis=0)
    rnk_s = rnk_w[rnk_w < 0].div(scalar, axis=0)
    weights = rnk_l.fillna(0).add(rnk_s.fillna(0), axis=1)
    return weights*0.5

def zsc(x, window):
    return (x - x.rolling(window).mean()) / (x.rolling(window).std()).replace(0, np.nan)

def BTester(weights, delay):
    capital = 100
    w = weights.shift(delay)
    w['RUB']['2022-02-24'::] = np.nan
    positions = round(capital * w)
    t_costs = abs((positions.diff() * slippage)).sum(axis=1)
    pnl = ((positions * np.log(TR.astype(float)).diff()).sum(axis=1) - t_costs)
    index = pnl.cumsum()
    strat_ret = (pnl / capital)
    strat_ret.mean() * 252
    sr = strat_ret.mean() / strat_ret.std() * np.sqrt(250)

    fx_pnl = ((positions * np.log(TR.astype(float)).diff()) - (abs(positions.diff()) * slippage))
    em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1).cumsum()
    dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1).cumsum()

    fx_rets = (fx_pnl/positions).fillna(0)

    index.plot(label='Strat')
    em_index.plot(label='EM')
    dm_index.plot(label='DM')
    plt.legend(loc='best')
    plt.show()
    print(round(sr, 2))

    df = pd.DataFrame(strat_ret, strat_ret.index, columns=['PNL'])
    df['year'] = df.index.year
    df['date'] = df.index.strftime('%m-%d')
    unstacked = df.set_index(['year', 'date']).PNL.unstack(-2)
    unstacked = unstacked.fillna(0)
    unstacked.cumsum()

    results = pd.DataFrame(index=unstacked.columns, columns=['CARG', 'Vol', 'SR', 'Omega', 'Sortino',
                                                             'Hit Ratio', 'RR', 'MDD', 'Calmar', 'AvgLoss',
                                                             'AvgGain', 'GainLoss', 'Stress10d', '1PctLoss',
                                                             '5PctLoss'])
    for i in unstacked.columns:
        index = (1 + unstacked[i]).cumprod()
        results.loc[i, 'CARG'] = ((index[-1] / 1) ** (1 / (len(index) / 360)) - 1) * 100
        results.loc[i, 'Vol'] = (unstacked[i].std() * np.sqrt(252)) * 100
        results.loc[i, 'SR'] = unstacked[i].mean() / unstacked[i].std() * np.sqrt(252)
        results.loc[i, 'Omega'] = np.sum(unstacked[i][unstacked[i] > 0]) / -np.sum(unstacked[i][unstacked[i] < 0])
        results.loc[i, 'Sortino'] = (unstacked[i].mean() * 250) / (unstacked[i][unstacked[i] < 0].std() * np.sqrt(250))
        results.loc[i, 'Hit Ratio'] = (unstacked[i][unstacked[i] > 0].count() / unstacked[i][
            unstacked[i] != 0].count()) * 100
        results.loc[i, 'RR'] = results.loc[i, 'Hit Ratio'] / (1 - results.loc[i, 'Hit Ratio'])
        results.loc[i, 'MDD'] = ((index / index.cummax() - 1).min()) * 100
        results.loc[i, 'Calmar'] = results.loc[i, 'CARG'] / -results.loc[i, 'MDD']
        results.loc[i, 'AvgGain'] = unstacked[i][unstacked[i] >= 0].mean() * 100
        results.loc[i, 'AvgLoss'] = unstacked[i][unstacked[i] < 0].mean() * 100
        results.loc[i, 'GainLoss'] = results.loc[i, 'AvgGain'] / -results.loc[i, 'AvgLoss']
        results.loc[i, 'Stress10d'] = unstacked[i].rolling(10).sum().min() * 100
        results.loc[i, '1PctLoss'] = unstacked[i][unstacked[i] < 0].quantile(0.01) * 100
        results.loc[i, '5PctLoss'] = unstacked[i][unstacked[i] < 0].quantile(0.05) * 100
    print(round(results.astype(float), 3))
    return weights, strat_ret.fillna(0), fx_rets

##

def StatArb(Px, long_level, short_level):
    # Define variables
    X = Ext_Factors[['S&P Mini Level', 'DXY Level', 'Oil Level']]
    X = np.log(X.astype(float)).diff()
    y = np.log(Px.astype(float)).diff()

    beta = []
    residuals = []
    for i in y:
        df = pd.concat([zsc(y[i], 60), zsc(X, 60)], axis=1).dropna()
        endog = df[i]
        exog = sm.add_constant(df.drop(i, axis=1))
        rols = RollingOLS(endog, exog, window=60, min_nobs=int(60 / 2))
        rres = rols.fit(params_only=True, reset=10)
        params = rres.params
        y_hat = params['const'] + params['S&P Mini Level'] * exog['S&P Mini Level'] + \
                params['DXY Level'] * exog['DXY Level'] + params['Oil Level'] * \
                exog['Oil Level']

        epsilon = (y[i] - y_hat).dropna()

        endog = epsilon
        exog = sm.add_constant(epsilon.shift(1))
        rols = RollingOLS(endog, exog, window=60, min_nobs=int(60 / 2))
        rres = rols.fit(params_only=True, reset=10)
        params = rres.params
        pred = params['const'] + params[0] * epsilon.shift(1)
        b = params[0]
        beta.append(b)
        res = (epsilon - pred)
        residuals.append(res)

    r = pd.concat(residuals, axis=1)
    r.columns = y.columns

    B = pd.concat(beta, axis=1)
    B.columns = y.columns

    signal = np.where(r >= long_level, -1, np.where(r <= short_level, 1, np.nan))
    signal = pd.DataFrame(signal, r.index, y.columns)
    signal = signal.reindex(y.index)
    return getweights(signal)


FXSTARB = StatArb(F3M, 1.75, -1.25)
weights1 = BTester(FXSTARB, 1)[0]

##
def MBO_Signal(px, period, barrier, hold):
    sma = px.rolling(period, min_periods=1, center=False).mean()
    max_px = px.rolling(period, min_periods=1, center=False).max()
    min_px = px.rolling(period, min_periods=1, center=False).min()
    MBO = (px - sma) / (max_px - min_px).replace(0, np.nan)
    signal = np.where(MBO >= barrier, -1, np.where(MBO <= -barrier, 1, np.nan))
    signal = pd.DataFrame(signal, px.index, px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

sf = np.log(EQC.astype(float)).diff() * 0.6 + np.log(FI.astype(float)).diff() * 0.4
sf_vol = 0.06 / (sf.ewm(90).std() * np.sqrt(252)).fillna(method='bfill')
sf_vol_adj = sf * sf_vol
sf_index = (1 + sf_vol_adj).cumprod()

EQCMR = MBO_Signal(sf_index, 105, 0.1, 5)
weights2 = BTester(EQCMR, 1)[0]

##
'Mean reversion on rates'

def TopBottom(x, window, n_assets):
    minmax = (x - x.rolling(window).max()) - (x.rolling(window).max() - x.rolling(window).min())
    signal = np.where(minmax.rank(ascending=True, axis=1) <= n_assets, 1,
                      np.where(minmax.rank(ascending=False, axis=1) <= n_assets, -1, 0))
    signal = pd.DataFrame(signal, x.index, x.columns).ewm(halflife=1).mean()
    return getweights(signal)

LTRMOM = TopBottom(Y5, 192, 3)
weights3 = BTester(LTRMOM, 1)[0]

##
'Fixed income to equities momentum'

def BinSignal(x, window):
    x = np.log(x.astype(float)).diff()
    mean = x.rolling(window).sum()
    signal = np.sign(mean)
    signal = pd.DataFrame(signal, x.index, x.columns)
    return getweights(signal)

EQCBIN = BinSignal(FI/EQC.replace(0, np.nan), 138)
weights4 = BTester(EQCBIN, 1)[0]

##
'Momentun on FX VaR breach'

def VAR_Shock(Px, level, window, hold):
    var = Px.rolling(window).quantile(level)
    signal = np.where(Px >= var.shift(1), 1, np.nan)
    signal = pd.DataFrame(signal, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

FXVARS = VAR_Shock(F3M, 0.9, 160, 2)
weights5 = BTester(FXVARS, 1)[0]

##
'Momentum on equities to fixed income'

def Alpha(px, level, period):
    df = zsc(px, period)
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

EQCT2 = Alpha(EQC/FI.replace(0, np.nan), 0.3, 130)
weights6 = BTester(EQCT2, 1)[0]

##
'Momentum with vol clustering and market regimes'

def TFollowing(px, n_assets, lb, lb1):
    rets = np.log(px.astype(float)).diff()
    vol = rets.ewm(14).std() * np.sqrt(252)
    condition = np.where(Ext_Factors['S&P Mini Level'].ewm(lb).mean() >= Ext_Factors['S&P Mini Level'].ewm(lb1).mean(),
                         1, -1)
    HVOL = pd.DataFrame(np.where(vol.rank(ascending=True, axis=1) <= n_assets,
                                 -1, 0), px.index, px.columns)
    LVOL = pd.DataFrame(np.where(vol.rank(ascending=False, axis=1) <= n_assets,
                                 -1, 0), px.index, px.columns)
    port = (HVOL.values + LVOL).mul(condition * -1, axis=0)

    return getweights(port)

SPTFO = TFollowing(FX, 4, 18, 837)
weights7 = BTester(SPTFO, 1)[0]

##
'Momentum with vol clustering and market regimes'

def TFollowing(px, n_assets, lb, lb1):
    rets = np.log(px.astype(float)).diff()
    vol = rets.ewm(14).std() * np.sqrt(252)

    asset = Ext_Factors['S&P Mini Level'] / Ext_Factors['DXY Level']
    condition = np.where(asset.ewm(lb).mean() >= asset.ewm(lb1).mean(), 1, -1)
    HVOL = pd.DataFrame(np.where(vol.rank(ascending=True, axis=1) <= n_assets,
                                 -1, 0), px.index, px.columns)
    LVOL = pd.DataFrame(np.where(vol.rank(ascending=False, axis=1) <= n_assets,
                                 -1, 0), px.index, px.columns)
    port = (HVOL.values + LVOL).mul(condition * -1, axis=0)
    return getweights(port)

EQUSD = TFollowing(F3M, 5, 70, 787)
weights8 = BTester(EQUSD, 1)[0]

##
'Mean reverting FX value'
import wbgapi as wb

Ctry = ['IND', 'MEX', 'CZE', 'ZAF', 'TUR', 'RUS', 'POL',
        'HUN', 'THA', 'SGP', 'CHN', 'DEU', 'AUS', 'NZL',
        'CAN', 'NOR', 'JPN', 'CHE', 'SWE', 'GBR', 'ROU', 'ISR']

ppp = wb.data.DataFrame('PA.NUS.PRVT.PP',
                        ['IND', 'MEX', 'CZE', 'ZAF', 'TUR', 'RUS', 'POL',
                         'HUN', 'THA', 'SGP', 'CHN', 'DEU', 'AUS', 'NZL',
                         'CAN', 'NOR', 'JPN', 'CHE', 'SWE', 'GBR', 'ROU', 'ISR'],
                        time=range(1992, 2022), numericTimeKeys=True,
                        index='time')

ppp_year = ppp[Ctry].shift(1).fillna(method='bfill')
ppp_year.columns = FX.columns
ppp_year.index = FX.resample('Y').mean().index
ppp_d = ppp_year.resample('D').last()
ppp_d = ppp_d.interpolate()
ppp_d = pd.DataFrame(ppp_d, FX.index)


def ValueStrat(Px, period, n_assets, hold):
    # Define variables
    y = np.log(Px.astype(float))
    X = np.log(ppp_d.astype(float))
    window = period

    results = []
    for i in y:
        df = pd.concat([y[i], X[i]], axis=1).dropna()
        endog = df.iloc[:, 0]
        exog = sm.add_constant(df.iloc[:, 1])
        rols = RollingOLS(endog, exog, window=window, min_nobs=int(window / 2))
        rres = rols.fit(params_only=True, reset=10)
        params = rres.params
        y_hat = params['const'] + params[i] * X[i]
        pred = y[i] - y_hat
        results.append(pred)
    residuals = pd.concat(results, axis=1, keys=FX.columns)
    signal = np.where(residuals.rank(ascending=True, axis=1) <= n_assets, 1,
                      np.where(residuals.rank(ascending=False, axis=1) <= n_assets, -1, np.nan))
    if hold == 0:
        signal = pd.DataFrame(signal, y.index, y.columns)
    else:
        signal = pd.DataFrame(signal, residuals.index, residuals.columns).fillna(method='ffill', limit=hold)

    return getweights(signal)

# Equity curve
FXVAL = ValueStrat(F3M, 2500, 2, 6)
weights9 = BTester(FXVAL, 1)[0]

##
'VaR breach on equity risk premium - momentum'

def VAR_Shock(x, window, hold):
    var = x.rolling(window).quantile(0.95)
    signal = np.where(x >= var.shift(1), 1, np.nan)
    signal = pd.DataFrame(signal, x.index, x.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

ERP = EY - Y10
FXVARS = VAR_Shock(ERP, 25, 1)
weights10 = BTester(FXVARS, 1)[0]

##
'Mean reversion in cash rates'

def Ratio(px, period, n_assets, hold):
    ratio = px / px.ewm(period).mean()
    signal = np.where(ratio.rank(ascending=False, axis=1) <= n_assets, 1,
                      np.where(ratio.rank(ascending=True, axis=1) <= n_assets, -1, np.nan))
    signal = pd.DataFrame(signal, px.index, px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

IRRATIO = Ratio(IR, 250, 3, 2)
weights11 = BTester(IRRATIO, 1)[0]

##
'Sentiment'

def VRP(window, long, short):
    window = window
    tot_returns = np.log(FX.astype(float)).diff()

    coef = pd.DataFrame(index=FX.iloc[59:].index, columns=FX.columns.tolist()).astype(float)
    epsilon = pd.DataFrame(index=FX.iloc[59:].index, columns=FX.columns.tolist()).astype(float)

    for i in tot_returns:
        y = tot_returns[i]
        X = (VOL1Y[i] - VOL1M[i]).astype(float).shift(1)
        df = pd.concat([X, y], axis=1, keys=['X', 'y']).dropna()
        X = df['X']
        y = df['y']

        for t in y.index[window - 1:]:
            y1 = y.loc[:t].iloc[-window:].values
            y1 = (y1 - y1.mean()) / y1.std()
            y1 = y1.reshape(-1, 1)

            X1 = X.loc[:t].iloc[-window:].values
            X1 = X1.reshape(-1, 1)

            model1 = LinearRegression(n_jobs=-1).fit(X1[:59], y1[:59])
            coef.loc[t, i] = model1.coef_
            epsilon.loc[t, i] = (y1[-1:] - model1.predict(X1[-1:]))

    coef = coef.reindex(FX.index)
    epsilon = epsilon.reindex(FX.index)

    signal = np.where(epsilon >= long, coef, np.where(epsilon <= short, coef, np.nan))
    signal = pd.DataFrame(signal, epsilon.index, epsilon.columns)
    return getweights(signal)

VRPRP = VRP(60, 3.5, -3.5)
weights13 = BTester(VRPRP, 0)[0]

##
'Momentum Disparity'

def Disparity(Px, Period, n_assets):
    Disp = (Px - Px.rolling(Period, min_periods=1).mean()) / Px.rolling(Period, min_periods=1).mean()
    signal = np.where(Disp.rank(ascending=True, axis=1) <= n_assets, 1,
                      np.where(Disp.rank(ascending=False, axis=1) <= n_assets, -1, 0))
    signal = pd.DataFrame(signal, Px.index, Px.columns)
    return getweights(signal)

IRDISP = Disparity(Y2, 188, 4)
weights14 = BTester(IRDISP, 1)[0]

##

'Momentum equities'

def Ribbon(Px, n_assets):
    dist = []
    for i in Px:
        rolling_means = {}
        for j in np.linspace(3, 500, 20):
            X = Px[i].rolling(window=int(j), center=False).mean()
            rolling_means[j] = X

        ribbons = pd.concat(rolling_means, axis=1)
        dist.append(ribbons.max(axis=1) - ribbons.min(axis=1))

    distance = pd.concat(dist, axis=1, keys=Px.columns)
    signal = np.where(distance.rank(ascending=True, axis=1) <= n_assets, -1,
                      np.where(distance.rank(ascending=False, axis=1) <= n_assets, 1, 0))

    signal = pd.DataFrame(signal, Px.index, Px.columns)
    return getweights(signal)


EQRIB = Ribbon(EQMR, 4)
weights16 = BTester(EQRIB, 1)[0]

##
'Mean reversionvalue in rates'

def RateValue(Rates, Period, n_assets, hold):
    real_rates = Rates - CPI.shift(20)
    Val = real_rates.ewm(halflife=Period, min_periods=1).mean()
    signal = np.where(Val.rank(ascending=True, axis=1) <= n_assets, -1,
                      np.where(Val.rank(ascending=False, axis=1) <= n_assets, 1, 0))
    signal= pd.DataFrame(signal, Rates.index, Rates.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)


RTVAL = RateValue(Y5, 56, 5, 1)
weights17 = BTester(RTVAL, 1)[0]

##
'IR equities'

def Ribbon(Px, n_assets):
    dist = []
    for i in Px:
        rolling_means = {}
        for j in np.linspace(3, 500, 20):
            X = Px[i].rolling(window=int(j), center=False).mean()
            rolling_means[j] = X

        ribbons = pd.concat(rolling_means, axis=1)
        euc_distance = np.sqrt((ribbons.max(axis=1) - ribbons.min(axis=1))**2)
        dist.append(euc_distance)

    distance = pd.concat(dist, axis=1, keys=Px.columns)
    signal = np.where(distance.rank(ascending=True, axis=1) <= n_assets, -distance,
                      np.where(distance.rank(ascending=False, axis=1) <= n_assets, distance, 0))

    signal = pd.DataFrame(signal, Px.index, Px.columns)
    return getweights(signal)

IRRIB = Ribbon(IRR, 5)
weights18 = BTester(IRRIB, 1)[0]

##
'FX momentum rank'

def rnk_mom(Px, Period, Level, hold):
    rets = np.log(Px.astype(float)).diff(Period)
    rets = rets.subtract(rets.mean(axis=1), axis=0)
    signal = np.where(rets.rank(ascending=True, axis=1, pct=True) > Level, 1,
                      np.where(rets.rank(ascending=True, axis=1, pct=True) < 1 - Level, -1, np.nan))
    signal = pd.DataFrame(signal, FX.index, FX.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

RNKMOM = rnk_mom(FI, 117, 0.99, 2)
weights19 = BTester(RNKMOM, 1)[0]

##
'Mean reversion in rates to equities'

def EQFICO(x, y, Period, Level):
    x, y = np.log(x.astype(float)).diff(), np.log(y.astype(float)).diff()
    rets = x.rolling(Period).corr(y)
    signal = np.where(rets.rank(ascending=True, axis=1, pct=True) > Level, -1,
                      np.where(rets.rank(ascending=True, axis=1, pct=True) < 1 - Level, 1, np.nan))
    signal = pd.DataFrame(signal, FX.index, FX.columns)
    return getweights(signal)

EQFIRO = EQFICO(Y10, EQC, 260, 0.20)
weights20 = BTester(EQFIRO, 1)[0]

##
'Momentum in vols ans riskies'

def VolSkew(x, y, Period, Level):
    x, y = (x.astype(float)).diff(), (y.astype(float)).diff()
    rets = x.rolling(Period).corr(y)
    signal = np.where(rets.rank(ascending=True, axis=1, pct=True) >= Level, 1,
                      np.where(rets.rank(ascending=True, axis=1, pct=True) <= 1 - Level, -1, np.nan))
    signal = pd.DataFrame(signal, FX.index, FX.columns)
    return getweights(signal)

RRRHO = VolSkew(VOL, RR3M, 136, 0.93)
weights21 = BTester(RRRHO, 1)[0]

##
'Mean reversion in equities to fixed income'

def IBS(Px, Period, Level, hold):
    ibs = (Px - Px.rolling(Period, min_periods=3).min()) / \
          (Px.rolling(Period, min_periods=3).max() - Px.rolling(Period, min_periods=3).min()).replace(0, np.nan)
    signal = np.where(ibs > Level, -1, np.where(ibs < 1 - Level, 1, np.nan))
    signal = pd.DataFrame(signal, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

FIMR2 = IBS(EQC/FI.replace(0, np.nan), 155, 0.65, 8)
weights22 = BTester(FIMR2, 1)[0]

##
'Momentum in equities to fixed income'

def MR(Px, lookback, std_dev, hold):
    ma = Px.ewm(halflife=lookback, min_periods=1).mean()
    std = Px.ewm(halflife=lookback, min_periods=1).std()
    upper_band = ma + std_dev * std
    lower_band = ma - std_dev * std
    signals = np.where(Px < lower_band, 1, np.where(Px > upper_band, -1, np.nan))
    signals = pd.DataFrame(signals, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signals)

EQFIMR = MR(EQC / FI.replace(0, np.nan), 12, 0.5, 2) * (10 / VOL)
weights23 = BTester(EQFIMR, 1)[0]

##
'Momentum in equities'

def mrs(Px, lookback, std_dev, hold):
    ma = Px.ewm(lookback, min_periods=1).mean()
    std = Px.ewm(lookback, min_periods=1).std()
    upper_band = ma + std_dev * std
    lower_band = ma - std_dev * std
    signals = np.where(Px < lower_band, 1, np.where(Px > upper_band, -1, np.nan))
    signals = pd.DataFrame(signals, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signals)

EQMR2 = mrs(EQC, 11, 2, 2) * (10 / VOL)
weights24 = BTester(EQMR2, 1)[0]

##

def alpha6(Px, lb, n_assets, hold):
    alpha = np.log(Px.astype(float)).diff(lb).rank(ascending=False, axis=0)
    signal = np.where(alpha.rank(ascending=True, axis=1) <= n_assets, 1,
                      np.where(alpha.rank(ascending=False, axis=1) <= n_assets, -1, np.nan))
    signal = pd.DataFrame(signal, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

EQFIMR = alpha6(FI/EQC, 154, 4, 2) * (10/VOL)
weights25 = BTester(EQFIMR, 1)[0]

##
def alpha7(Px, lb):
    returns = np.log(Px.astype(float)).diff()
    mean = returns.ewm(halflife=lb, min_periods=int(lb / 2)).mean()
    std = returns.ewm(halflife=lb, min_periods=int(lb / 2)).std()
    tstat = (np.sqrt(lb) * mean) / std
    signal = 2 * norm.cdf(tstat) - 1
    weights = getweights(pd.DataFrame(-signal, Px.index, Px.columns))
    return weights

EQTF = alpha7(EQM/IR.replace(0, np.nan), 103)
weights26 = BTester(EQTF, 1)[0]

##

def MRalpha(Px, window, long, short):
    z_score = zsc(Px, window)
    signal = np.where(z_score >= long, -1, np.where(z_score <= short, 1, np.nan))
    signal = pd.DataFrame(signal, Px.index, Px.columns)
    return getweights(signal)

MRFXTR = MRalpha(TOT, 94, 2.95, -2.16)
weights29 = BTester(MRFXTR, 1)[0]

##
# TOT mean reversion
def MRXS(df, period, long, short, hold):
    returns = df.astype(float).diff(period)
    z_score = returns.subtract(returns.mean(axis=1), axis=0).div(returns.std(axis=1), axis=0)
    signal = np.where(z_score >= long, -1, np.where(z_score <= short, 1, np.nan))
    signal = pd.DataFrame(signal, df.index, df.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

MRTOT = MRXS(TOT, 13, 2.5,-2,1)*(10/VOL)
weights30 = BTester(MRTOT, 1)[0]

##

def CarAdj():
    car = ((F3M-FX)/F3M)*360
    rnk = car.rank(ascending=True, axis=1)
    rnk_w = rnk.subtract(rnk.mean(axis=1), axis=0)/100
    return rnk_w

CARADJ = CarAdj()
weights31 = BTester(CARADJ, 1)[0]

##
def GrowthAlpha(Px1, Px2, Period1, Period2):
    zsc1 = zsc(Px1, Period1)
    rnk1 = zsc1.rank(ascending=False, axis=1)
    zsc2 = zsc(Px2, Period2)
    rnk2 = zsc2.rank(ascending=False, axis=1)
    rnk = rnk1*0.5 + rnk2*0.5
    rnk_w = rnk.subtract(rnk.mean(axis=1), axis=0)/100
    return rnk_w.fillna(0)

GROWTH = GrowthAlpha(EQC/IR.replace(0, np.nan), TOT, 30, 80)
weights32 = BTester(GROWTH, 1)[0]

##
def Efi_Ratio(Px, period, level):
    n = period
    change = Px.diff(n).abs()
    vol = Px.diff().abs().rolling(n).sum()
    ER = change / vol.replace(0, np.nan)
    norm = (ER-ER.mean())/ER.std()
    signals = np.where(norm >= level, 1, np.where(norm <= -level, -1, 0))
    return getweights(pd.DataFrame(signals, FX.index, FX.columns))

FIER = Efi_Ratio(FI, 38, 1.5)
weights33 = BTester(FIER, 1)[0]

##

TOTER = Efi_Ratio(TOTR, 32, 2)
weights34 = BTester(TOTER, 1)[0]

##

def CCI(Close, High, Low, period, level):
    n = period
    TP = (High + Low + Close)/3
    SMA_TP = TP.rolling(n).mean()
    SMA_STD = TP.rolling(n).std()
    cci_ind = (TP - SMA_TP) / (0.015*SMA_STD).replace(0, np.nan)
    norm = (cci_ind-cci_ind.mean())/cci_ind.std()
    signals = np.where(norm >= level, -1, np.where(norm <= -level, 1, 0))
    signals = pd.DataFrame(signals, FX.index, FX.columns)
    return getweights(signals)

EQCCI = CCI(EQC, EQH, EQL, 55, 1.75)
weights34 = BTester(EQCCI, 1)[0]

##

def ProbMoM(Px, lb):
    returns = np.log(Px.astype(float)).diff()
    mean = returns.ewm(halflife=lb, min_periods=int(lb / 2)).mean()
    std = returns.ewm(halflife=lb, min_periods=int(lb / 2)).std()
    tstat = (np.sqrt(lb) * mean) / std
    signal = 2 * norm.cdf(tstat) - 1
    weights = getweights(pd.DataFrame(-signal, Px.index, Px.columns))
    return weights

PMOM = ProbMoM(PER/IRR.replace(0, np.nan), 600)
weights35 = BTester(PMOM, 1)[0]

##

def Alpha(px, level, period):
    zsc = (px - px.rolling(period, min_periods=int(period/2)).mean()) / \
         (px.rolling(period, min_periods=int(period/2)).std()).replace(0, np.nan)
    signal = np.where(zsc.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(zsc.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

PBMOM = Alpha(PB, 0.4, 50)
weights37 = BTester(PBMOM, 1)[0]

##
def Alpha(px, level, period):
    zsc = (px - px.rolling(period, min_periods=int(period/2)).mean()) /\
          (px.rolling(period, min_periods=int(period/2)).std()).replace(0, np.nan)
    signal = np.where(zsc.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(zsc.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

DCAR['RON'] = np.nan
DIRMOM = Alpha(DCAR, 0.4, 12)
weights38 = BTester(DIRMOM, 1)[0]

##
def Alpha(px, level, period):
    df = px.astype(float).pct_change(period)
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

LCAR['ILS'] = np.nan
LCAR['RON'] = np.nan
LCARMOM = Alpha(LCAR/IRR.replace(0,np.nan), 0.1, 140)
weights39 = BTester(LCARMOM, 1)[0]

##
def Alpha(px, level, period):
    df = px/px.ewm(halflife=period, min_periods=int(period/2)).mean().replace(0,np.nan)
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, -1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, 1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

RELFIX = Alpha(FIR, 0.2, 100)
weights40 = BTester(RELFIX, 1)[0]

##
def Alpha(px, level, period):
    df = px/px.ewm(halflife=period, min_periods=int(period/2)).mean().replace(0,np.nan)
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

EQRFIR = Alpha(EQCR/FIR.replace(0, np.nan), 0.3, 27)
weights41 = BTester(EQRFIR, 1)[0]

##
def Alpha(px, level, period):
    df = px/px.ewm(halflife=period, min_periods=int(period/2)).mean().replace(0,np.nan)
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

EQVALRC = Alpha(PER/IRR.replace(0, np.nan), 0.3, 250)
weights42 = BTester(EQVALRC, 1)[0]

##
def alpha013(Close, Volume, Period):
    vol_rnk = Volume.rank(pct=True)
    close_rnk = Close.rank(pct=True)
    cov = close_rnk.rolling(Period).cov(vol_rnk)
    return getweights(cov.rank(pct=True))

EQVOLU = alpha013(EQC, EQV, 160)
weights43 = BTester(EQVOLU, 1)[0]

##
def Alpha(px, level, period):
    df = px.pct_change().ewm(halflife=period, min_periods=int(period/4)).mean()
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, px.index, px.columns))

EQMOM2 = Alpha(EQC/IR.replace(0,np.nan), 0.3, 24)
weights44 = BTester(EQMOM2, 1)[0]

##

def Alpha(Rate, level, period):
    risk_factor = Ext_Factors['S&P Mini']
    covar = Rate.diff().rolling(period, min_periods=int(period/4)).cov(risk_factor)
    var = risk_factor.rolling(period, min_periods=int(period/4)).var()
    beta = covar.div(var, axis=0)
    df = Rate*beta
    signal = np.where(df.rank(ascending=True, pct=True, axis=1) <= level, 1,
                      np.where(df.rank(ascending=False, pct=True, axis=1) <= level, -1, np.nan))
    return getweights(pd.DataFrame(signal, Rate.index, Rate.columns))

MODCAR = Alpha(S3MR, 0.1, 225)
weights45 = BTester(MODCAR, 1)[0]

##
def Alphai(Px, level, window, hold):
    z = zsc(Px, window).astype(float)
    z = z.subtract(z.mean(axis=1), axis=0)
    sigmoid = (1/(1 + np.exp(-z)))
    signal = np.where(sigmoid <= level, 1, np.nan)
    signal = pd.DataFrame(signal, Px.index, Px.columns).fillna(method='ffill', limit=hold)
    return getweights(signal)

FXWINS = Alphai(F3M, 0.15, 240, 1)
weights46 = BTester(FXWINS, 1)[0]

##
fwrd = (S6M*0.5 - S3M*0.25)/0.25
CARSIGM = Alphai(fwrd, 0.2, 220, 1)
weights47 = BTester(CARSIGM, 1)[0]

##
def Alphaii(Px, window):
    z = zsc(Px, window).astype(float)
    signal = np.tanh(z)
    return getweights(-signal)

sfp = (np.log(EQC.astype(float)).diff() * 0.6 + np.log(FI.astype(float)).diff() * 0.4)
sfpi = (1+sfp).cumprod()
SFTFM = Alphaii(sfpi/IR.replace(0,np.nan), 120)
weights48 = BTester(SFTFM, 1)[0]

##
def usdir(Px, window, level):
    ef = Ext_Factors['DXY']
    rets = np.log(Px.astype(float)).diff()
    z = zsc(ef, window)
    condition = np.where(z > level, 1, np.nan)
    new_df = rets.mul(condition, axis=0)
    signal = np.where(new_df.rank(ascending=True, pct=True, axis=1) <= 0.4, 1, 0)
    return getweights(pd.DataFrame(signal, Px.index, Px.columns))

USDDIR = usdir(F3M, 284, 1.2)
weights49 = BTester(USDDIR, 1)[0]

##
def EqMoM3(Px, window):
    rets = np.log(Px.astype(float)).diff(window)
    rets_dm = rets.subtract(np.nanmean(rets, 1), axis=0)
    signals = -np.sign(rets_dm.astype(float))
    signals['RUB']['2022-02-24'::] = np.nan
    return getweights(pd.DataFrame(signals, Px.index, Px.columns))

EQMOM3 = EqMoM3(EQC/FI.replace(0,np.nan), 102)
weights50 = BTester(EQMOM3, 1)[0]

##
def HedgeCarry(Px, period):
    window = period
    X = Ext_Factors['S&P Mini'].astype(float)
    y = np.log(FX.astype(float)).diff()
    beta = y.rolling(window).cov(X).div(X.rolling(window).var(), axis=0)*100
    car = Px.ewm(halflife=period).mean()/beta
    return -rank_w(car)


HEDGEDC = HedgeCarry((((F3M-FX)/F3M)*360), 260)
weights51 = BTester(HEDGEDC, 1)[0]

##

tot_returns = np.log(TR.astype(float)).diff()

Ccy = FX.columns.tolist()

rets = np.log(TR.astype(float)).diff().fillna(0)

df = pd.concat([weights2,weights9,weights5,weights23,weights14,
                weights20,weights21,weights16,weights10,weights25,
                weights6,weights14,weights32,weights34,weights30,
                weights33,weights37,weights38,weights40,weights41,
                weights42,weights43,weights44,weights45,weights46,
                weights47,weights48,weights49,rets],
               keys=['f1', 'f2', 'f3', 'f4','f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                     'f11', 'f12', 'f13', 'f14','f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                     'f21', 'f22', 'f23', 'f24','f25', 'f26', 'f27', 'f28', 'rets'],
               axis=1)


# Calendar Features
cal = pd.DataFrame(index=TR.index)
cal['DoW'] = np.sin(2 * np.pi * TR.index.weekday/5.0)
cal['week'] = np.sin(2 * np.pi * TR.index.day/30.0)
cal['month'] = np.sin(2 * np.pi * TR.index.month/12.0)

alphas = pd.concat([np.sign(alphas.replace(0,np.nan))], axis=1)
y = tot_returns.fillna(0)
X = alphas.fillna(0)
X = X.reindex(y.index)

##

def SignalCombiner(window_size):
    window = window_size
    fcst = pd.DataFrame(index=df.index[window-1:], columns=Ccy)

    for t in df.index[window-1:]:
        # Random subsample
        X = df.drop('rets', axis=1).loc[:t].iloc[-window:].shift(1)
        y = df['rets'].loc[:t].iloc[-window:]

        y = (y-y.mean())/y.std()
        X = (X-X.mean())/X.std()

        X_train = X[:-3].fillna(0)
        y_train = y[:-3].fillna(0)

        X_test = X[-3:].fillna(0)
        y_test = y[-3:].fillna(0)

        model = Lasso(alpha=0.01)
        model.fit(X_train, y_train)
        pred_train = pd.DataFrame(model.predict(X_train), y_train.index, y_train.columns)
        pred_test =  pd.DataFrame(model.predict(X_test), y_test.index, y_test.columns)

        opt_weights = Optimizer(pred_train)

        fcst.loc[t, Ccy] = model.predict(X[-1:].fillna(0).stack())

    weights = fcst.replace(0,np.nan).drop(['RUB', 'INR'], axis=1)
    rnk = weights.rank(ascending=True, axis=1)
    rnk_w = rnk.subtract(rnk.mean(axis=1), axis=0) / 100
    return rnk_w

%time rnk_w = SignalCombiner(1000)

##

# Equity curve
capital = 8500
margin = 0
leverage = 1
exposure = capital * (1 - margin) * leverage
w = rnk_w
positions = round(exposure * w)
tot_returns = np.log(TR.astype(float)).diff().drop(['RUB', 'INR'], axis=1)
on_returns = np.log(FX.astype(float)) - np.log(FX_Open.astype(float))

costs = slippage.drop(['RUB', 'INR'], axis=1)
fx_pnl = (positions * tot_returns) - (abs(positions.diff()) * costs)

cost_fx = abs((positions.diff()) * costs*1.5)
t_costs = cost_fx.sum(axis=1)
pnl = ((positions * tot_returns).sum(axis=1) - t_costs).replace(0, np.nan)
em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1).cumsum()
dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1).cumsum()

strat_ret = (pnl / capital)
sr = (strat_ret.replace(0, np.nan).mean() / strat_ret.replace(0, np.nan).std() * np.sqrt(250))
index = (1 + strat_ret.replace(0, np.nan)).cumprod().dropna()

mmd = np.max((np.maximum.accumulate(index) - index) / np.maximum.accumulate(index))
roc = strat_ret.mean()*252
carg = (index[-1] / 1) ** (1 / (index.index.year[-1]-index.index.year[0])) - 1

p = np.where(np.sign(positions.replace(0,np.nan)) == np.sign(positions.replace(0,np.nan)).shift(1), 0, 1)
turnover_ratio = round(np.mean(np.sum(p, 1)/p.shape[1]),1)
pt = ((abs(w[w>0].diff())/0.5).sum(axis=1) +(abs(w[w<0].diff())/0.5).sum(axis=1)).replace(0,np.nan).mean()

vol = strat_ret.std() * np.sqrt(252)
avg_ret = strat_ret.mean() * 250
sortino = (strat_ret.mean() * 250) / (strat_ret[strat_ret < 0].std() * np.sqrt(250))
hit_ratio = pnl[pnl > 0].count() / pnl[pnl != 0].count()
risk_reward = hit_ratio / (1 - hit_ratio)
calmar = carg / mmd
omega = np.sum(strat_ret[strat_ret > 0]) / -np.sum(strat_ret[strat_ret < 0])
max_stress_10_day = strat_ret.rolling(10).sum().min()
top_1pct_loss = strat_ret[strat_ret < 0].quantile(0.01)
top_5pct_loss = strat_ret[strat_ret < 0].quantile(0.05)
usd_exp = positions[positions > 0].sum(axis=1) - abs(positions[positions < 0]).sum(axis=1)

avg_gain = pnl[pnl >= 0].mean()
avg_loss = pnl[pnl <= 0].mean()
gain_loss = avg_gain / -avg_loss
skew = pnl.skew()
kurt = pnl.kurt()
t_stat = pnl.mean() / pnl.std() * np.sqrt(250 * (len(pnl) / 360))
var = strat_ret.quantile(0.05)
cvar_95 = strat_ret[strat_ret <= var].mean()

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(pnl.cumsum(), label='strat')
ax[0].plot(dm_index, label='DM')
ax[0].plot(em_index, label='EM')
ax[0].legend(loc='best')
ax[0].set_title('Strat Compounded Performance', fontweight='bold')
ax[0].set_ylabel('Equity Curve')
ax[0].grid(True, lw=0.3)
ax[1].plot(fx_pnl.cumsum())
ax[1].legend(fx_pnl.columns, loc='best', bbox_to_anchor=(1.03, 1.03))
ax[1].set_title('PNL by Asset', fontweight='bold')
ax[1].set_ylabel('Cumulative Returns')
ax[1].grid(True, lw=0.3)
plt.tight_layout()
plt.show()

table = [[carg * 100, vol * 100, sr, mmd * 100, hit_ratio * 100,
          calmar, sortino, omega, var * 100, cvar_95 * 100, max_stress_10_day * 100,
          top_1pct_loss * 100, top_5pct_loss * 100, turnover_ratio]]

headers = ['CARG%', 'Vol%', 'SR', 'MDD%', 'HR%', 'Calmar%',
           'Sortino', 'Omega', 'VAR95%', 'CVAR95%', 'Stress 10d',
           'Top 1% Loss', 'Top 5% Loss', 'Turnover']

print(tabulate(table, headers, tablefmt="github", numalign="center", floatfmt=".2f"))

print('-----------')

##
'Risk Adj'

unleveraged_rets = strat_ret.copy()
# Equity curve
margin = 0.2
roll_vol = unleveraged_rets.ewm(21).std() * np.sqrt(252)
vol_target = 0.5
leverage = vol_target / roll_vol
exposure = capital * (1 - margin) * round(leverage)
positions = round(w.mul(exposure, axis=0))
positions = positions.clip(upper=40000).clip(lower=-40000)
turnover = positions.rolling(252).corr(positions.shift(1)).mean(axis=1)

fx_pnl = (positions * tot_returns) - (abs(positions.diff()) * costs).replace(0, np.nan)
fx_pnl_pct = fx_pnl / positions.replace(0, np.nan)
cost_fx = abs((positions.diff()) * costs*1.5)
t_costs = cost_fx.sum(axis=1)
pnl = (positions * tot_returns).sum(axis=1) - t_costs
pnl = pnl.replace(0, np.nan)

em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1).cumsum()
dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1).cumsum()

strat_ret = (pnl / capital).replace(0, np.nan)
sr = (strat_ret.mean() / strat_ret.std() * np.sqrt(250))
index = (1 + strat_ret).cumprod().dropna()
mmd = np.max((np.maximum.accumulate(index) - index) / np.maximum.accumulate(index))
carg = (index[-1] / 1) ** (1 / (index.index.year[-1]-index.index.year[0])) - 1

p = np.where(np.sign(positions) == np.sign(positions).shift(1), 0, 1)
turnover_ratio = np.mean(np.sum(p, 1)/p.shape[1])

vol = strat_ret.std() * np.sqrt(252)
avg_ret = strat_ret.mean() * 250
sortino = (strat_ret.mean() * 250) / (strat_ret[strat_ret < 0].std() * np.sqrt(250))
hit_ratio = pnl[pnl > 0].count() / pnl[pnl != 0].count()
risk_reward = hit_ratio / (1 - hit_ratio)
calmar = carg / mmd
omega = np.sum(strat_ret[strat_ret > 0]) / -np.sum(strat_ret[strat_ret < 0])
max_stress_10_day = strat_ret.rolling(10).sum().min()
top_1pct_loss = strat_ret[strat_ret < 0].quantile(0.01)
top_5pct_loss = strat_ret[strat_ret < 0].quantile(0.05)
usd_exp = positions[positions > 0].sum(axis=1) - abs(positions[positions < 0]).sum(axis=1)

avg_gain = pnl[pnl >= 0].mean()
avg_loss = pnl[pnl <= 0].mean()
gain_loss = avg_gain / -avg_loss
skew = pnl.skew()
kurt = pnl.kurt()
t_stat = pnl.mean() / pnl.std() * np.sqrt(250 * (len(pnl) / 360))
var = strat_ret.quantile(0.05)
cvar_95 = strat_ret[strat_ret <= var].mean()

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(pnl.cumsum(), label='strat')
ax[0].plot(dm_index, label='DM')
ax[0].plot(em_index, label='EM')
ax[0].legend(loc='best')
ax[0].set_title('Strat Compounded Performance', fontweight='bold')
ax[0].set_ylabel('Equity Curve')
ax[0].grid(True, lw=0.3)
ax[1].plot(fx_pnl.cumsum())
ax[1].set_title('PNL by Asset', fontweight='bold')
ax[1].set_ylabel('Cumulative Returns')
ax[1].grid(True, lw=0.3)
plt.show()

table = [[carg * 100, vol * 100, sr, mmd * 100, hit_ratio * 100,
          calmar, sortino, omega, var * 100, cvar_95 * 100, max_stress_10_day * 100,
          top_1pct_loss * 100, top_5pct_loss * 100, turnover_ratio]]

headers = ['CARG%', 'Vol%', 'SR', 'MDD%', 'HR%', 'Calmar%',
           'Sortino', 'Omega', 'VAR95%', 'CVAR95%', 'Stress 10d',
           'Top 1% Loss', 'Top 5% Loss', 'Turnover']

print(tabulate(table, headers, tablefmt="github", numalign="center", floatfmt=".2f"))

print('-----------')

summary = pd.concat([round(positions[-3:]), FX[-3:],
            round(fx_pnl[-3:])], axis=0).T.drop(['INR', 'RUB'], axis=0)

print(summary)

abs(positions).sum(axis=1)

strat_ret.resample('Y').sum().plot(kind='bar')
plt.show()

pnl['2022-08'].cumsum().plot(), plt.show()
