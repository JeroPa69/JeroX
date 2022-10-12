import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import classification_report


def bt_stats(pnl):
    pnl = pnl.dropna()
    ret = pnl[pnl != 0].mean()*250
    vol = pnl[pnl != 0].std()*np.sqrt(250)
    
    index = (1+pnl).cumprod()
    mmd = (index/index.cummax()-1).min()
    
    sr_htp = ret/vol
    sortino = ret/(pnl[pnl<0].std()*np.sqrt(250))
    hit_ratio = pnl[pnl>=0].count()/pnl[pnl != 0].count()
    risk_reward = hit_ratio/(1-hit_ratio)
    calmar = ret/-mmd
    
    avg_gain = pnl[pnl >= 0].mean()
    avg_loss = pnl[pnl <= 0].mean()
    
    skew = pnl.skew()
    kurt = pnl.kurt()    
    t_stat = pnl.mean()/pnl.std()*np.sqrt(250*(len(pnl)/360))
    
    var = -np.percentile(pnl, 5)
    cvar_95 = pnl[pnl >= var].mean()
    
    print('--------Stats--------')
    print('Return p.a.', round(ret*100, 2))
    print('Volatility p.a.', round(vol*100, 2))
    print('Sharpe Ratio', round(sr_htp, 2))
    print('Sortino', round(sortino, 2)) 
    print('Calmar', round(calmar, 2)) 
    print('Hit Ratio', round(hit_ratio, 2))
    print('Risk/Reward', round(risk_reward, 2)) 
    print('Skew', round(skew, 2))
    print('Maximum DD', round(mmd, 3))
    print('VAR95',round(var, 3))
    print('CVAR95',round(cvar_95, 3))
    print('T.Stat',round(t_stat, 3))

    return bt_stats

def zsc(x, window):
    return (x-x.rolling(window).mean()) / (x.rolling(window).std()).replace(0, np.nan)

def tstat(x, window):
    return x.pct_change().rolling(window).mean()/x.pct_change().rolling(window).std()*np.sqrt(window)

def minmax(x, window):
    return (x-x.rolling(window).min()) / (x.rolling(window).max()-x.rolling(window).min()).replace(0, np.nan)

def robust_scaler(x, window):
    return (x-x.rolling(window).median())/(x.rolling(window).quantile(0.75)-x.rolling(window).quantile(0.25))

def roll_beta(X, y, window):
    covar = X.pct_change().rolling(window).cov(y.pct_change())
    var = y.pct_change().rolling(window).var()
    beta = covar.div(var, axis=0)
    return beta

def roll_beta_abs(X, y, window):
    covar = X.pct_change().rolling(window).cov(y)
    var = y.rolling(window).var()
    beta = covar.div(var, axis=0)
    return beta

roll_beta_abs(S3M, Ext_Factors['S&P Mini Level'], 90)

covar = S3M.diff().rolling(90).cov(Ext_Factors['S&P Mini'])
var = Ext_Factors['S&P Mini'].rolling(90).var()
beta = covar.div(var, axis=0)



#------------------------------------------------------------

def label(x):
    return pd.DataFrame(np.where(x > 0, 1, 0), x.index, x.columns)

def sign(x, window):
    return np.sign(x.pct_change(window))
    
def binary(x, window):
    return pd.DataFrame(np.where(x.pct_change(window) < 0, 1, 0), x.index, x.columns)

def tanh(x):
    t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return 1-t**2










##

def getweights(signals):
    long = signals[signals > 0].div(signals[signals > 0].sum(axis=1), axis=0)
    short = signals[signals < 0].div(signals[signals < 0].sum(axis=1), axis=0)
    return (long.fillna(0) - short.fillna(0))/2

#---------------------------------------------------

def ATR(Spot, High, Low, period):
    atr = []
    for i in Spot:
        H_L = High[i]-Low[i]
        H_C = abs(High[i] - Spot[i].shift(1))
        L_C = abs(Low[i] - Spot[i].shift(1))
        
        ranges = pd.concat([H_L, H_C, L_C], axis=1)
        atr.append(ranges.max(axis=1))
    
    ATR = pd.concat(atr, axis=1, keys=Spot.columns)
    
    return ATR.rolling(period).sum()/period


def StopLoss(signals, ATR_level, period):
    
    ATR_long_USD =  Spot - ATR_level * ATR(Spot, High, Low, period)
    
    ATR_short_USD =  Spot + ATR_level * ATR(Spot, High, Low, period)
    
    filtered_signal = np.where(Spot < ATR_long_USD.shift(1), 0, signals)
    
    final_signal = np.where(Spot > ATR_short_USD.shift(1), 0, filtered_signal)
    
    return pd.DataFrame(final_signal, signals.index, signals.columns)


#---------------------------------------------------

def GetStats(weights):
    capital = 100
    weights = getweights(weights)
    positions = round(capital * weights.shift(1)).fillna(0)
    t_costs = abs((positions.diff()) * full_tc).sum(axis=1)['1998'::]
    pnl = ((positions * np.log(TR.astype(float)).diff()).sum(axis=1)-t_costs)['1998'::]
    index = pnl.cumsum()
    strat_ret = (index.diff()/capital)
    sr = strat_ret.mean()/strat_ret.std()*np.sqrt(250)

    ret = index.diff()/capital
    avg_ret = ret.mean()*250
    vol = np.std(ret)*np.sqrt(250)
    sr_htp = avg_ret/vol

    mmd = (index-index.cummax()).min()

    sortino = pnl.mean()/(ret[ret < 0].std()*np.sqrt(250))
    hit_ratio = pnl[pnl > 0].count()/pnl[pnl != 0].count()
    risk_reward = hit_ratio/(1-hit_ratio)
    calmar = (avg_ret*capital)/-mmd
    
    avg_gain = pnl[pnl >= 0].mean()
    avg_loss = pnl[pnl <= 0].mean()
    
    skew = pnl.skew()
    t_stat = pnl.mean()/pnl.std()*np.sqrt(250*(len(pnl)/360))
    
    var = -np.percentile(pnl, 5)
    cvar_95 = pnl[pnl >= var].mean()

    print('--------Stats--------')
    print('Return p.a.', round(avg_ret*100, 2))
    print('Volatility p.a.', round(vol*100, 2))
    print('SR', round(sr_htp, 2))
    print('Sortino', round(sortino, 2))
    print('Calmar', round(calmar, 2))
    print('Hit Ratio', round(hit_ratio, 2))
    print('Risk/Reward', round(risk_reward, 2))
    print('Avg. Gain', round(avg_gain, 2))
    print('Avg. Loss', round(avg_loss, 2))
    print('Skew', round(skew, 2))
    print('MDD (USD)', round(mmd, 3))
    print('VAR95',round(var, 3))
    print('CVAR95',round(cvar_95, 3))
    print('T.Stat',round(t_stat, 3))

    return


#-----------------------------------------------------------------


def bt_abs(Name, signal, capital, TR, costs, Spot, df_volas, FX_Open, IR,
           High, Low, leverage, stop_loss_ATR, stop_loss_period,
           vol_adj_signal: bool = True, smooth_signal: bool = True):
    
    if vol_adj_signal == True and smooth_signal == True:
        signals = signal.ewm(span=2, adjust=True).mean() * (10/df_volas)
                   
    if vol_adj_signal == False and smooth_signal == True:
        signals = signal.ewm(span=2, adjust=True).mean()

    if vol_adj_signal == True and smooth_signal == False:
        signals = signal * (10/df_volas)
    else:
        signals = signal.copy(deep=True)
        
    #-----------------------------------------------------------------------
    
    def ATR(Spot, High, Low, stop_loss_period):
        atr = []
        for i in Spot:
            H_L = High[i]-Low[i]
            H_C = abs(High[i] - Spot[i].shift(1))
            L_C = abs(Low[i] - Spot[i].shift(1))
            
            ranges = pd.concat([H_L, H_C, L_C], axis=1)
            atr.append(ranges.max(axis=1))
        
        ATR = pd.concat(atr, axis=1, keys=Spot.columns)
        
        return ATR.rolling(stop_loss_period).sum()/stop_loss_period


    def StopLoss(signals, ATR_level, stop_loss_period):
        
        ATR_long_USD =  Spot - ATR_level * ATR(Spot, High, Low, stop_loss_period)
        
        ATR_short_USD =  Spot + ATR_level * ATR(Spot, High, Low, stop_loss_period)
        
        filtered_signal = np.where(Spot < ATR_long_USD.shift(1), 0, signals)
        
        final_signal = np.where(Spot > ATR_short_USD.shift(1), 0, filtered_signal)
        
        return pd.DataFrame(final_signal, signals.index, signals.columns)
    
    #-----------------------------------------------------------------------

    returns = np.log(TR.astype(float)).diff()
    signal = StopLoss(signal, stop_loss_ATR, stop_loss_period)
    weights = getweights(signal)
    positions = round(capital * weights.shift(1)).fillna(0)
    t_costs = (abs(positions.diff()) * costs).sum(axis=1)['1993'::]

    #-----------------------------------------------------------------------
    

    ldn_spot = np.log(FX_Open.astype(float)).diff()
    FX_carry = (np.log(TR.astype(float)).diff() - np.log(Spot.astype(float)).diff())
    ldn_ret = (ldn_spot + FX_carry).replace([np.inf, -np.inf], 0)

    pnl = ((positions * leverage * returns).sum(axis=1)-t_costs)['1993'::]
    pnl_ldn = ((positions * leverage * ldn_ret).sum(axis=1)-t_costs)['1993'::]
    pnl_half = ((positions * leverage * ldn_ret).sum(axis=1)-t_costs*0.5)['1993'::]
    pnl_nocost = (positions * leverage * returns).sum(axis=1)['1993'::]
    
    ccy_pnl = (positions * leverage * returns) - (abs(positions.diff()) * costs)
    ccy_pct = ccy_pnl.cumsum().diff() / capital

    em_pnl = (positions * leverage * ldn_ret).iloc[:, 0:11].sum(axis=1)-t_costs
    dm_pnl = (positions * leverage * ldn_ret).iloc[:, 11:20].sum(axis=1)-t_costs

    index = pnl.cumsum().replace([np.inf, -np.inf], 0)
    index_ldn = pnl_ldn.cumsum().replace([np.inf, -np.inf], 0)
    index3 = pnl_half.cumsum().replace([np.inf, -np.inf], 0)
    
    y_rets = (index.diff()/capital).resample('Y').sum()
    
    #------------------------------------------------------------------
    
    ret = index.diff()/capital
    avg_ret = ret.mean()*250
    vol = np.std(ret)*np.sqrt(250)
    sr_htp = avg_ret/vol

    ret3 = index3.diff()/capital
    avg_ret3 = ret3.mean()*250
    vol3 = np.std(ret3)*np.sqrt(250)
    sr3 = avg_ret3/vol3

    ret_ldn = index_ldn.diff()/capital
    avg_ret_ldn = np.mean(ret_ldn)*250
    vol_ldn = np.std(ret_ldn)*np.sqrt(250)
    sr_ldn = avg_ret_ldn/vol_ldn

    mmd = (index-index.cummax()).min()

    sortino = (ret.mean()*250)/(ret[ret < 0].std()*np.sqrt(250))
    hit_ratio = pnl[pnl > 0].count()/pnl[pnl != 0].count()
    risk_reward = hit_ratio/(1-hit_ratio)
    calmar = (avg_ret*capital)/-mmd
    
    avg_gain = pnl[pnl >= 0].mean()
    avg_loss = pnl[pnl <= 0].mean()
    gain_loss = avg_gain/-avg_loss
    
    skew = pnl.skew()
    kurt = pnl.kurt()
    t_stat = pnl.mean()/pnl.std()*np.sqrt(250*(len(pnl)/360))
    
    var = -np.percentile(pnl, 5)
    cvar_95 = pnl[pnl >= var].mean()
    
    y_pred = np.sign(positions)
    y_true = np.sign(returns)
    
    true_pos = np.where((y_pred == 1) & (y_true == 1), 1, 0).sum() / y_pred[y_pred == 1].count().sum()
    false_pos = np.where((y_pred == 1) & (y_true != 1), 1, 0).sum() /  y_pred[y_pred == 1].count().sum()
    true_negs = np.where((y_pred == -1) & (y_true == -1), 1, 0).sum() /  y_pred[y_pred == -1].count().sum()
    false_negs = np.where((y_pred == -1) & (y_true != -1), 1, 0).sum() /  y_pred[y_pred == -1].count().sum()
    
    print('--------Stats--------')
    print('Return p.a.', round(avg_ret*100, 2))
    print('Volatility p.a.', round(vol*100, 2))
    print('SR NY', round(sr_htp, 2))
    print('SR Ldn Open', round(sr_ldn, 2))
    print('SR Half Cost', round(sr3, 2))
    print('Sortino', round(sortino, 2))
    print('Calmar', round(calmar, 2))
    print('Hit Ratio', round(hit_ratio, 2))
    print('Risk/Reward', round(risk_reward, 2))
    print('Avg. Gain', round(avg_gain, 2))
    print('Avg. Loss', round(avg_loss, 2))
    print('Skew', round(skew, 2))
    print('Kurt', round(kurt, 2))
    print('MDD (USD)', round(mmd, 3))
    print('VAR95',round(var, 3))
    print('CVAR95',round(cvar_95, 3))
    print('T.Stat',round(t_stat, 3))
    print(' ')
    print('--Signals Evaluation--')
    print('True Long USD',round(true_pos, 2))
    print('False Long USD',round(false_pos, 2))
    print('True Short USD',round(true_negs, 2))
    print('False Short USD',round(false_negs, 2))
    
    stats = [[avg_ret*100, vol*100, sr_htp, sr_ldn, sr3,
                       sortino, calmar, hit_ratio, risk_reward, avg_gain,
                       avg_loss, skew, kurt, mmd, var, cvar_95, t_stat,
                       true_pos, false_pos, true_negs, false_negs]]
    
    row_labels = ['Return p.a.', 'Volatility p.a.', 'SR NYT', 'SR LDNT',
                    'SR Half Cost', 'Sortino', 'Calmar', 'Hit Ratio',
                    'Risk-Reward', 'Avg. Gain', 'Avg. Loss', 'Skew',
                    'Kurt', 'MDD (USD)', 'VAR95', 'CVAR95', 'T.Stat',
                    'True USD Long', 'False USD Long', 'True USD Short',
                    'False USD Short']
    
    #------------------------------------------------------------------
    fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6],
         [ax7, ax8]) = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
    fig.tight_layout(h_pad=2)
    
    fig.suptitle('Analitical Summary', fontweight='bold', fontsize=22)
    fig.subplots_adjust(top=0.88)
    
    ax1.plot(index.dropna(), label='NY Net PNL')
    ax1.plot(index_ldn.dropna(), label='Ldn Net PNL')
    ax1.plot(pnl_half.dropna().cumsum(), label='Ldn PNL Costs x 0.5')
    ax1.plot(pnl_nocost.dropna().cumsum(), label='NY Gross PNL')
    ax1.plot(t_costs.dropna().cumsum(), label='Cum Costs')
    ax1.set_title(Name, fontweight='bold', fontsize=16)
    ax1.set_ylabel('Equity Curve', fontweight='bold')
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, lw=0.1)

    ax2.plot(pnl.index, ((positions * ldn_spot).sum(axis=1)-t_costs).dropna().cumsum(), label='Spot PNL')
    ax2.plot(pnl.index, ((positions * FX_carry).sum(axis=1)-t_costs).dropna().cumsum(), label='Carry PNL')
    ax2.plot(dm_pnl.index, dm_pnl.cumsum(), label='G10 PNL')
    ax2.plot(em_pnl.index, em_pnl.cumsum(), label='EM PNL')
    ax2.plot(ccy_pnl.drop(ccy_pnl.cumsum()[-1:].idxmax(axis=1), axis=1).sum(axis=1).cumsum(), label='PNL wo Top')
    ax2.plot(ccy_pnl.drop(ccy_pnl.cumsum()[-1:].idxmin(axis=1), axis=1).sum(axis=1).cumsum(), label='PNL wo Bottom')
 
    ax2.set_ylabel('Equity Curve', fontweight='bold')
    ax2.set_title('PNL Breakdown Carry/Spot', fontweight='bold', fontsize=16)
    ax2.legend(loc='upper left')
    ax2.grid(True, lw=0.1)
    
    ax3.plot(pnl.rolling(90).mean()/pnl.rolling(90).std()*np.sqrt(252), label='Rolling SR')
    ax3.set_ylabel('Sharpe Ratio p.a.', fontweight='bold')
    ax3.set_title('90d Rolling SR', fontweight='bold', fontsize=16)
    ax3.legend(loc='best')
    ax3.grid(True, lw=0.1)

    ax4.plot(round((index/index.rolling(250, min_periods=1).max()-1)['2002'::], 2), label='Daily MMD')
    ax4.set_ylabel('In %.', fontweight='bold')
    ax4.set_title('Max Daily DD', fontweight='bold', fontsize=16)
    ax4.legend(loc='best')
    ax4.grid(True, lw=0.1)

    (index.diff()/capital).hist(bins=50, ax=ax5, density=True, edgecolor='white')
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax5.set_title('Daily Net Returns Histogram', fontweight='bold', fontsize=16)
    ax5.set_xlim([-0.03, 0.03])
    ax5.set_xticks([-0.03, -0.02, 0.01, 0, 0.01, 0.02, 0.03])
    ax5.grid(True, lw=0.1)
    
    ax6.bar(y_rets.index, y_rets, width=200, label='Net Returns (in %)')
    ax6.set_title('Yearly Net Returns', fontweight='bold', fontsize=16)
    ax6.set_ylabel('Returns (in %)', fontweight='bold', fontsize=12)
    ax6.grid(True, lw=0.1)
    
    ax7.plot(ccy_pnl.cumsum(),  label=ccy_pnl.columns)
    ax7.set_title('Performance by Currency', fontweight='bold', fontsize=16)
    ax7.legend(loc='best', ncol=4)
    ax7.set_ylabel('Returns (in %)', fontweight='bold', fontsize=12)
    ax7.grid(True, lw=0.1)
    
    image = plt.imshow(ccy_pnl.corr(), interpolation='nearest', aspect="auto")
    plt.colorbar(image)
    ax8.set_xticklabels(Spot.columns.tolist(), rotation=90)
    ax8.set_yticklabels(Spot.columns.tolist())
    ax8.set_title('FX PNL Correlation', fontweight='bold', fontsize=16)
    
    ax7.plot(ccy_pnl.cumsum())
    ax7.set_title('Performance by Currency', fontweight='bold', fontsize=16)
    #ax7.legend(loc='best', ncol=4)
    ax7.set_ylabel('Returns (in %)', fontweight='bold', fontsize=12)
    ax7.grid(True, lw=0.1)

    plt.draw()

    plt.tight_layout(h_pad=2)

    #-----------------------------------------------------------------

    return
