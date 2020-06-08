# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:31:50 2020

@author: Vostok
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'GSPC' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'GSPC')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'GSPC':  # drop dates SPY did not trade
            df = df.dropna(subset=["GSPC"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    dr = df.copy()
    dr = (df / df.shift(1) - 1)
    dr.iloc[0] = 0  # Torna a a primeira linha igual a zaro para não estourar os cálculos
    return dr
    # Note: Returned DataFrame must have the same number of rows



# Read data
dates = pd.date_range('1995-06-01', '2020-06-01')  # one month only
symbols = ['GSPC', 'BOV', 'BRL']
df = get_data(symbols, dates)
df['BOV'].fillna(method='ffill', inplace=True)
df['BRL'].fillna(method='ffill', inplace=True)
df['USDBOV'] = df['BOV'] / df['BRL']
df_norm = df/df.iloc[0]

plt.figure(figsize=(20,10),dpi=300)
plt.title("Índices Normalizados do S&P 500 versus IBOV em USD")
plt.plot(df_norm[['BOV','USDBOV','GSPC']],linewidth =1)
plt.legend(['IBOV Normalizado','IBOV em USD Normalizado','S&P500 Normalizado'],loc='lower right')
plt.show()


# Compute daily returns
daily_returns = compute_daily_returns(df)

print('Correlação')
print(daily_returns[['GSPC','BOV','BRL','USDBOV']].corr(method='pearson'))
print()
print('Curtose')
print(daily_returns.kurtosis())

mean = daily_returns.mean()
std = daily_returns.std()


daily_returns.hist(['USDBOV'],bins=100)
plt.title("Histograma dos Retornos Diários do IBOV em USD")
plt.axvline(-std['GSPC'], color='r', linestyle='dashed', linewidth=2)
plt.axvline(+std['GSPC'], color='r', linestyle='dashed', linewidth=2)
plt.axvline(mean['GSPC'], color='w', linestyle='dashed', linewidth=2)
plt.show()



plt.figure(figsize=(20,10),dpi=150)
plt.title("Retornos Diários do IBOV em USD")
plt.plot(daily_returns['USDBOV'],linewidth=0.2)
plt.show()

plt.figure(figsize=(20,10),dpi=150)
plt.title("Retornos Diários do IBOV em USD versus S&P 500")
plt.plot(daily_returns['GSPC'], daily_returns['USDBOV'], '.')
beta_BOV, alpha_BOV = np.polyfit(daily_returns.GSPC, daily_returns.BOV, 1)
plt.plot(daily_returns['GSPC'], beta_BOV * daily_returns['GSPC'] + alpha_BOV, '-r')
plt.show()