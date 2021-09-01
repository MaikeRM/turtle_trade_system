import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from copy import deepcopy, copy
import turtlesystem as TurtleSystem
from functions import getStratStats

# Sample 10 tickers from S&P 500
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table = pd.read_html(url)
df = table[0]
syms = df['Symbol']
# Sample symbols
tickers = list(np.random.choice(syms.values, size=10))
print("Ticker Symbols:")
_ = [print(f"\t{i}") for i in tickers]
sys = TurtleSystem(tickers, init_account_size=1E4, start='2000-01-01')
sys.run()


port_values = sys.get_portfolio_values()
returns = port_values / port_values.shift(1)
log_returns = np.log(returns)
cum_rets = log_returns.cumsum()

# Compare to SPY baseline
sp500 = yf.Ticker('SPY').history(start=sys.start, end=sys.end)
sp500['returns'] = sp500['Close'] / sp500['Close'].shift(1)
sp500['log_returns'] = np.log(sp500['returns'])
sp500['cum_rets'] = sp500['log_returns'].cumsum()

plt.figure(figsize=(12, 8))
plt.plot((np.exp(cum_rets) -1 )* 100, label='Turtle Strategy')
plt.plot((np.exp(sp500['cum_rets']) - 1) * 100, label='SPY')
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.title('Cumulative Portfolio Returns')
plt.legend()
plt.tight_layout()
plt.show()

stats = getStratStats(log_returns)
spy_stats = getStratStats(sp500['log_returns'])
df_stats = pd.DataFrame(stats, index=['Turtle'])
df_stats = pd.concat([df_stats, pd.DataFrame(spy_stats, index=['SPY'])])
df_stats