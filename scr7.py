##
import numpy as np
import pandas as pd

def simulate_px(Px, n_vols, freq, sample_size=100):
    # Historical returns
    returns = np.log(Px.astype(float)).diff().dropna()
    # Block sample returns to preserve AUC, tails and X-sectional statistics
    rnd_returns = [returns.iloc[x:x+sample_size] for x in np.random.randint(len(returns), size=1)][0]
    # Block sample annualized vols
    vol = rnd_returns.std()*np.sqrt(252) * n_vols
    # Simulate Geometric Brownian Motion
    rnd_norm = pd.DataFrame(np.random.normal(0, 1, size=(Px.shape[0], Px.shape[1])), Px.index, Px.columns)
    # Perform Cholesky decomposition on coefficient matrix
    rnd_corr = rnd_returns.corr().interpolate()
    R = np.linalg.cholesky(rnd_corr)
    epsilon = np.inner(rnd_norm, R)
    dt = 1 / freq
    noise = (r/100 - 0.5 * v**2) * dt + vol * np.sqrt(dt) * pd.DataFrame(epsilon, Px.index, Px.columns)
    noise = np.exp(noise.astype(float))
    # Generate prices with noise
    sim_px = Px.ewm(span=3).mean() * noise
    return sim_px, noise



