import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from copy import deepcopy, copy
import turtlesystem as TurtleSystem
from functions import getStratStats

# Sample 10 tickers from S&P 500
url = 'https://pt.wikipedia.org/wiki/Lista_de_companhias_citadas_no_Ibovespa'
table = pd.read_html(url)
df = table[0]
syms = np.empty(10, dtype = object)
i=0
for acao in df['Código']:
    syms[i] = (acao+'.SA')
    i+=1
    if i == 10: break
#syms = np.asarray(syms)
# Sample symbols
tickers = list(np.random.choice(syms, size=10))
#tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'JBSS3.SA', 'FLRY3.SA', 'ABEV3.SA']
print("Ticker Symbols:")
_ = [print(f"\t{i}") for i in tickers]
sys = TurtleSystem(tickers, init_account_size=1E4, start='2013-01-01')
sys.run()

port_values = sys.get_portfolio_values()
returns = port_values / port_values.shift(1)
log_returns = np.log(returns)
cum_rets = log_returns.cumsum()

# Compare to SPY baseline
BVSP = yf.Ticker('^BVSP').history(start=sys.start, end=sys.end)
BVSP['returns'] = BVSP['Close'] / BVSP['Close'].shift(1)
BVSP['log_returns'] = np.log(BVSP['returns'])
BVSP['cum_rets'] = BVSP['log_returns'].cumsum()

plt.figure(figsize=(12, 8))
plt.plot((np.exp(cum_rets) -1 )* 100, label='Turtle Strategy')
plt.plot((np.exp(BVSP['cum_rets']) - 1) * 100, label='BVSP')
plt.xlabel('Data')
plt.ylabel('Returno (%)')
plt.title('Retorno acumulado do portifólio')
plt.legend()
plt.tight_layout()
plt.show()

stats = getStratStats(log_returns)
bvsp_stats = getStratStats(BVSP['log_returns'])
df_stats = pd.DataFrame(stats, index=['Turtle'])
df_stats = pd.concat([df_stats, pd.DataFrame(bvsp_stats, index=['BVSP'])])
df_stats