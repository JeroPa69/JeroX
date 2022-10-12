##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import optuna

def Sig(Px, period):
    window = period
    X = Ext_Factors['S&P Mini'].astype(float)
    y = np.log(FX.astype(float)).diff()
    beta = y.rolling(window).cov(X).div(X.rolling(window).var(), axis=0)*100
    car = Px.ewm(halflife=period).mean()/beta
    w = rank_w(car)
    return w

def objective(trial):
    x = trial.suggest_int("x", 3, 250)

    capital = 100
    w = -Sig((((F3M-FX)/F3M)*360), 260)
    weights = w.shift(1)
    positions = round(capital * weights)
    t_costs = (abs(positions.diff() * slippage)).sum(axis=1)
    pnl = (positions * np.log(TR.astype(float)).diff()).sum(axis=1) - t_costs
    index = pnl.cumsum()
    strat_ret = (pnl / capital)
    sr = strat_ret.mean() / strat_ret.std() * np.sqrt(250)
    tratio = np.nanmean(strat_ret) / np.nanstd(strat_ret)* np.sqrt(250) * (len(strat_ret.dropna())/250)**0.5
    pvalue = scipy.stats.t.sf(abs(tratio), df=12)
    pm = 1-(1-pvalue)**100

    fx_pnl = ((positions * np.log(TR.astype(float)).diff()) - (abs(positions.diff()) * slippage))
    em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1).cumsum()
    dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1).cumsum()
    sharpes = fx_pnl.mean()/fx_pnl.std()*np.sqrt(252)
    ratio = np.sign(sharpes).replace(-1, 0).sum()/len(sharpes)

    fx_pnl.cumsum().plot(label='Strat')
    index.plot(label='EM')
    plt.legend(loc='best')
    plt.show()
    print(round(ratio, 2))
    return round(tratio, 2)

if __name__ == "__main__":
    # Let us minimize the objective function above.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=200.0)
    df = study.trials_dataframe()
    print(df.sort_values('value', ascending=False))
    print(study.best_trial.params.items())
