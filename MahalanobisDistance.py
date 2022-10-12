from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

#create function to calculate Mahalanobis distance

def mahalanobis(x):
    x_mu = x - np.mean(x)
    cov = np.cov(x.astype(np.float64).T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return pd.DataFrame(mahal.diagonal(), x.index, columns=['Mahala_Dist'])

df = Ext_Factors[['S&P Mini', 'Ind Metals', 'SemiCond', 'MSCI EM', 'Kospi', 'Nikei',
                  'Vix', 'Move', 'Gold', 'Copper', 'Silver', 'US OAS', 'UST10Y',
                  'US2Y', 'DXY', 'Vdax']]

regime = mahalanobis(df.ewm(halflife=5).mean().fillna(0))


reg = pd.DataFrame(index=FX.index[251:], columns=FX.columns.tolist()).astype(float)

regimes = []
for i in FX:
    Ccy = 'PLN'
    df = pd.concat([EQC[Ccy].pct_change(90), VOL1Y[Ccy].diff(), TOTR[Ccy].diff(),
                    FI[Ccy].pct_change(90), Y10[Ccy]-Y2[Ccy], F3M[Ccy]/FX[Ccy],
                    FX_Open[Ccy]/FX[Ccy].shift(-1)-1], axis=1).dropna()

    for t in df.index[251:]:
            tmp_df = df.loc[:t]
            tmp_df = tmp_df.iloc[-252:]
            maha = mahalanobis(tmp_df)
            regimes.append(maha[-1:].values)

pd.concat(regimes, axis=1)

reg.plot()
plt.show()


#create new column in dataframe that contains Mahalanobis distance for each row
df['mahalanobis'] = mahalanobis(x=df, data=df[['MXN=X', 'CZK=X', 'ZAR=X', 'RUB=X', 'PLN=X','HUF=X', 'THB=X',
           'SGD=X', 'CNY=X', 'EUR=X', 'AUD=X', 'NZD=X',
           'CAD=X', 'NOK=X', 'JPY=X', 'CHF=X', 'SEK=X',
           'GBP=X']])

df['mahalanobis'].plot()


#calculate p-value for each mahalanobis distance 
df['p'] = 1 - chi2.cdf(df['mahalanobis'], 3)

df['mahalanobis'].plot()

regime = df[df['mahalanobis'].between(19, 21)].mean()
today = df[-1:].T




def RegSignal(period, n_assets):
    
    df = VOL.copy()
    names = df.columns.tolist()

    #create function to calculate Mahalanobis distance
    def mahalanobis(x=None, data=None):
        x_mu = x - np.mean(data)
        cov = np.cov(data.astype(np.float64).T)
        cov
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left, x_mu.T)
        return mahal.diagonal()

    #create new column in dataframe that contains Mahalanobis distance for each row
    df['mahalanobis'] = mahalanobis(x=df, data=df[['BRL',
     'MXN',
     'CZK',
     'ZAR',
     'TRY',
     'RUB',
     'PLN',
     'HUF',
     'THB',
     'SGD',
     'CNH',
     'EUR',
     'AUD',
     'NZD',
     'CAD',
     'NOK',
     'JPY',
     'CHF',
     'SEK',
     'GBP']])

    # Define variables
    y = np.log(TR.astype(float)).diff()
    X = df['mahalanobis']

    # Run rolling reg per currency (250 days)
    results = []
    for i in y:
        endog = y[i].dropna()
        exog = sm.add_constant(X[-len(endog):].fillna(method='bfill'))
        rols = RollingOLS(endog, exog, window=period, min_nobs=int(period/2))
        rres = rols.fit(params_only=True, reset=10)
        params = rres.params
        y_hat = params.iloc[:,0] + params.iloc[:,1] * exog.iloc[:,1]
        results.append(y_hat)

    
    betas = pd.concat(results, axis=1, keys=FX.columns)

    signal = np.where(betas.rank(ascending=True, axis=1) <= n_assets, 1, 0)
                      
    return pd.DataFrame(signal, betas.index, betas.columns)



# Equity curve
REGRP = RegSignal(305, 2)
capital = 100
weights1 = getweights(REGRP.shift(1))
positions = round(capital * weights1).fillna(0)
t_costs = abs((positions.diff() * full_tc)).sum(axis=1)
pnl = ((positions * np.log(TR.astype(float)).diff()).fillna(0).sum(axis=1)- t_costs)['1998'::]
index = pnl.cumsum()
strat_ret = (index.diff()/capital)
strat_ret.mean()*252
sr = strat_ret.mean()/strat_ret.std()*np.sqrt(250)
pnl.cumsum().plot()
print(round(sr, 2))

bt_abs('Backtesting Full System', REGRP, 100, TR, full_tc, FX, VOL, FX_Open, IR,
       H, L, leverage = 3, stop_loss_ATR = 2, stop_loss_period = 11,
       vol_adj_signal=False, smooth_signal=False)


def objective(trial):
    x = trial.suggest_int("x", 1, 1000)
    y = trial.suggest_int("y", 1, 5)

    capital = 100
    
    signal = RegSignal(x, y)
    weights = getweights(signal)
    positions = round(capital * weights.shift(1)).fillna(0)
    t_costs = abs((positions.diff() * full_tc)).sum(axis=1)['1998'::]
    pnl = ((positions * np.log(TR.astype(float)).diff()).sum(axis=1)-t_costs)['1998'::].dropna()
    index = pnl.cumsum()
    strat_ret = (index.diff()/capital).fillna(0)['2004':'2018']
    sr = (strat_ret.mean()/strat_ret.std()*np.sqrt(250))
    sortino =  (strat_ret.mean()*252)/(strat_ret[strat_ret < 0].std()*np.sqrt(252))
    gain_loss = strat_ret[strat_ret >= 0].mean() / -strat_ret[strat_ret <= 0].mean()
    hit_ratio = strat_ret[strat_ret > 0].count()/strat_ret[strat_ret != 0].count()
    smooth = -((strat_ret.diff())**2).sum()*100
    pnl.cumsum().plot()
    print(round(sr, 2))

    return sr

if __name__ == "__main__":
    # Let us minimize the objective function above.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=50.0)
    df = study.trials_dataframe()
    print(df.sort_values('value', ascending=False))




