##
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

'Calculate Net Alpha Returns'

alphas = [weights1,weights2,weights3,weights5,weights6,weights9,
          weights10,weights16,weights14,weights20,weights21,
          weights23,weights25,weights30,weights32,weights33,
          weights34,weights37,weights38,weights40,weights41,
          weights42,weights43,weights44,weights45,weights46,
          weights47,weights48,weights49,weights51]

net_pnls = []
for i in range(len(alphas)):
    adj_alphas = (alphas[i].shift(1).replace(0, np.nan)).fillna(0)
    capital = 100
    returns = np.log(TR.astype(float)).diff()
    weights = getweights(adj_alphas)
    positions = round(capital * weights)
    t_costs = abs((positions.diff()) * slippage).sum(axis=1)
    pnl = ((positions * returns).sum(axis=1) - t_costs)
    strat_ret = (pnl / capital)
    net_pnls.append(strat_ret)

all_pnls = pd.concat(net_pnls, axis=1)
alpha_cumrets = all_pnls.cumsum()
alpha_cumrets.plot(), plt.show()

alpha_returns = all_pnls - all_pnls.mean(axis=0)

def HRP_Optimizer(alpha_returns):

    estimate_correl = alpha_returns.corr(method='pearson')
    estimate_covar = pd.DataFrame(LedoitWolf().fit(alpha_returns).covariance_) #shrinkage
    distances = np.sqrt((1 - estimate_correl) / 2)

    def seriation(Z, N, cur_index):
        """Returns the order implied by a hierarchical tree (dendrogram)."""
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return (seriation(Z, N, left) + seriation(Z, N, right))


    def serial_matrix(dist_mat, method="ward"):
        """Returns a sorted distance matrix"""
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method=method)
        res_order = seriation(res_linkage, N, N + N - 2)
        seriated_dist = np.zeros((N, N))
        a,b = np.triu_indices(N, k=1)
        seriated_dist[a,b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
        seriated_dist[b,a] = seriated_dist[a,b]

        return seriated_dist, res_order, res_linkage

    def HRP_weights(covariances, res_order):
        weights = pd.Series(1, index=res_order)
        clustered_alphas = [res_order]

        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                   (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                left_subcovar = covariances[left_cluster].loc[left_cluster]
                inv_diag = 1 / np.diag(left_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                right_subcovar = covariances[right_cluster].loc[right_cluster]
                inv_diag = 1 / np.diag(right_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor

        return weights

    ordered_dist_mat, res_order, res_linkage = serial_matrix(distances.values, method='single')

    return pd.DataFrame(HRP_weights(estimate_covar, res_order))

HRP_Optimizer(all_pnls)

W = HRP_Optimizer(all_pnls).T

weighted_alphas = 0
n = 0
for i in W:
    adj_alphas = pd.DataFrame(W[i].values * alphas[i].values, alphas[i].index)
    weighted_alphas += adj_alphas
    n += 1

weighted_alphas.columns = FX.columns

weighted_alphas['MXN'].loc['1/1/1993':'31/3/1995'] = np.nan
weighted_alphas['TRY'].loc['1/1/1993':'1/1/2001'] = np.nan
weighted_alphas['CNH'].loc['1/1/1993':'1/1/2006'] = np.nan
weighted_alphas['RUB'].loc['1/1/1993':'1/1/2009'] = np.nan
weighted_alphas['THB'].loc['1/1/1993':'3/3/2008'] = np.nan
weighted_alphas['INR'].loc['1/1/1993':'1/1/2010'] = np.nan

capital = 100
margin = 0
leverage = 1
vol_adj_alphas = weighted_alphas
exposure = capital * (1 - margin) * leverage
weights = getweights(vol_adj_alphas.drop(['RUB', 'RON', 'INR'], axis=1))
positions = round(exposure * weights * 5/VOL)

fx_pnl = (positions * np.log(TR.astype(float)).diff().drop(['RUB', 'RON', 'INR'], axis=1)) - \
         (abs(positions.diff()) * slippage.drop(['RUB', 'RON', 'INR'], axis=1))

t_costs = abs((positions.diff()) * slippage.drop(['RUB', 'RON', 'INR'], axis=1)).sum(axis=1)

pnl = ((positions * np.log(TR.astype(float)).diff().drop(['RUB', 'RON', 'INR'], axis=1)).sum(axis=1) - t_costs)

em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1).cumsum()
dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1).cumsum()
strat_ret.describe()
strat_ret = pnl / capital
sr = (strat_ret.mean() / strat_ret.std() * np.sqrt(250))
index = (1 + strat_ret).cumprod()
mmd = np.max((np.maximum.accumulate(index) - index) / np.maximum.accumulate(index))
carg = (index[-1] / 1) ** (1 / (len(index) / 360)) - 1

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
ax[0].set_title('Aggregated Strat Performance', fontweight='bold')
ax[0].set_ylabel('Equity Curve')
ax[0].grid(True, lw=0.3)
ax[1].plot(fx_pnl.cumsum())
ax[1].set_title('PNL by Traded Asset', fontweight='bold')
ax[1].set_ylabel('Cumulative Returns')
ax[1].grid(True, lw=0.3)
plt.show()

table = [[carg * 100, vol * 100, sr, mmd * 100, hit_ratio * 100,
          calmar, sortino, omega, var * 100, cvar_95 * 100, max_stress_10_day * 100,
          top_1pct_loss * 100, top_5pct_loss * 100]]

headers = ['CARG%', 'Vol%', 'SR', 'MDD%', 'HR%', 'Calmar%',
           'Sortino', 'Omega', 'VAR95%', 'CVAR95%', 'Stress 10d',
           'Top 1% Loss', 'Top 5% Loss']

print(tabulate(table, headers, tablefmt="github", numalign="center", floatfmt=".2f"))


unleveraged_rets = strat_ret.copy()
# Equity curve
capital = 5000
margin = 0.2
roll_vol = unleveraged_rets.ewm(14).std()*np.sqrt(252)
vol_target = 0.3
leverage = vol_target/roll_vol
exposure = capital * (1 - margin) * leverage
positions = round(weights.mul(exposure, axis=0))
positions = positions.clip(upper=30000).clip(lower=-30000)

fx_pnl = ((positions * np.log(TR.astype(float)).diff().drop(['RUB', 'INR', 'RON'], axis=1)) - \
          (abs(positions.diff()) * slippage.drop(['RUB', 'INR', 'RON'], axis=1)))


t_costs = abs((positions.diff()) * slippage.drop(['RUB', 'INR', 'RON'], axis=1)).sum(axis=1)

pnl = ((positions * np.log(TR.astype(float)).diff().drop(['RUB', 'INR', 'RON'], axis=1)).sum(axis=1) - t_costs)

em_index = fx_pnl.drop(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK'], axis=1).sum(axis=1)
dm_index = (fx_pnl[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'NZD', 'CHF', 'SEK', 'NOK']]).sum(axis=1)

pnl_bd = pd.concat([fx_pnl.sum(axis=1), em_index, dm_index], axis=1, keys=['PNL', 'EM', 'DM']).fillna(0).resample('M').sum()

fx_pnl['2022-05'].cumsum().plot(subplots='True', kind='bar', figsize=(20,20), color='#e6e6ff'), plt.show()

pnl_bd.resample('Y').sum()

pd.concat([pnl, em_index.diff(), dm_index.diff()], axis=1, keys=['PNL', 'EM', 'DM']).fillna(0).resample('Y').sum()

pnl_bd.cumsum().plot(), plt.show()

strat_ret = (pnl / capital)

sr = (strat_ret.mean() / strat_ret.std() * np.sqrt(250))
index = (1 + strat_ret).cumprod()
mmd = np.max((np.maximum.accumulate(index) - index) / np.maximum.accumulate(index))
carg = (index[-1] / 1) ** (1 / (len(index) / 360)) - 1

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
ax[0].plot(dm_index.cumsum(), label='DM')
ax[0].plot(em_index.cumsum(), label='EM')
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
          top_1pct_loss * 100, top_5pct_loss * 100]]

headers = ['CARG%', 'Vol%', 'SR', 'MDD%', 'HR%', 'Calmar%',
           'Sortino', 'Omega', 'VAR95%', 'CVAR95%', 'Stress 10d',
           'Top 1% Loss', 'Top 5% Loss']

print(tabulate(table, headers, tablefmt="github", numalign="center", floatfmt=".2f"))

pd.concat([positions.tail(2), FX.tail(2),
           round(fx_pnl.tail(3))], axis=0).T.drop(['INR', 'RUB', 'RON'], axis=0)

##

((strat_ret['2022']).cumsum()*100).plot(c='r')
plt.grid(True, lw=0.3)
plt.title('PNL%', fontweight='bold')
plt.ylabel('Cumulative Performance (EOD)', fontweight='bold')
plt.xlabel(' ')
plt.tight_layout()
plt.show()

for i in strat_ret.resample('y').first().index.year:
    ((strat_ret[str(i)]).cumsum()*100).plot(c='r')
    plt.grid(True, lw=0.3)
    plt.title('PNL%', fontweight='bold')
    plt.ylabel('Cumulative Performance (EOD)', fontweight='bold')
    plt.xlabel(' ')
    plt.tight_layout()
    plt.show()


sim = []
for i in range(3000):
    sim.append(strat_ret[np.random.randint(5000, size= 250)].reset_index(drop='True'))

(pd.concat(sim, axis=1)).cumsum().plot(lw=0.6)
(pd.concat(sim, axis=1)).mean(axis=1).cumsum().plot(c='purple')
plt.legend().remove()
plt.title('S&P Bear Market Regimes Comparison', fontweight='bold')
plt.ylabel('Compounded Returns', fontweight='bold')
plt.show()
