import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from envs.backtest.base import Candles, BacktestSingle, COMMISSION
from functools import partial
from gym import spaces
from tqdm import tqdm
import warnings

# ignore runtime warnings from computing the sortino ratio
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ignore runtime warnings for chained assignment in pandas
pd.options.mode.chained_assignment = None


class BacktestIndicator(BacktestSingle):
    def __init__(self, datapath=None, strategy=None, **kwargs):
        super().__init__(datapath=datapath, **kwargs)
        # Transform OHLC to CULR
        self.data = pd.DataFrame(
            None,
            index=self.raw.index,
            columns=['Close', 'Upper', 'Lower', 'Real', 'Volume']
        )
        self.data['Close'] = self.raw['Close'].copy()
        self.data['Upper'] = self.raw['High'] - self.raw[['Open', 'Close']].max(axis=1)
        self.data['Lower'] = self.raw[['Open', 'Close']].min(axis=1) - self.raw['Low']
        self.data['Real'] = self.raw['Close'] - self.raw['Open']
        self.data['Volume'] = self.raw['Volume'].copy()

        # Gramian Angular Summation Field
        self.transformer = GramianAngularField()

        self.bt = partial(self.bt, strategy=strategy)
        self.reset()

        self.observation_space = spaces.Box(
            -np.ones((5, Candles.train, Candles.train), dtype=np.float32),
            np.ones((5, Candles.train, Candles.train), dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = strategy.action_space

    def preprocess(self, data):
        return self.transformer.transform(data.T)

    def reset(self):
        if self.training:
            len_data = (len(self.data)
                        - (Candles.train + Candles.val + Candles.test))
            t = np.random.randint(len_data)
        else:
            self.t = t = len(self.data) - Candles.train - Candles.test - 1
            self.trades = None
            self.pbar = tqdm(total=Candles.test)
        self.obs = self.data[t:t + Candles.train + Candles.val]
        return self.preprocess(self.obs[:Candles.train].copy())

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.training:
            return self.train_step(action)
        else:
            return self.eval_step(action)

    def train_step(self, action):
        start = self.obs.index[Candles.train]
        end = self.obs.index[-1]
        bt = self.bt(data=self.raw.loc[self.obs.index[0]:end])

        kwargs = {'n' + str(i + 1): action[i] for i in range(len(action))}
        equity = bt.run(**kwargs)['_equity_curve']['Equity']
        exp_return = equity[end] / (equity[start] + 1e-8) - 1
        reward = exp_return * 100  # give reward by percentage (%)

        kwargs = {}
        for i in range(len(action)):
            key = 'n' + str(i + 1)
            kwargs[key] = getattr(bt._strategy, key + '_base')
        equity = bt.run(**kwargs)['_equity_curve']['Equity']
        baseline = equity[end] / (equity[start] + 1e-8) - 1

        obs = self.reset()
        done = False
        info = {
            'return': exp_return,
            'bnh': self.raw['Close'][end] / self.raw['Open'][start] - 1,
            'baseline': baseline,
        }
        return obs, reward, done, info

    def eval_step(self, action):
        start = self.obs.index[Candles.train]
        end = self.obs.index[-1]
        bt = self.bt(data=self.raw.loc[self.obs.index[0]:end])

        kwargs = {'n' + str(i + 1): action[i] for i in range(len(action))}
        trades = bt.run(**kwargs)['_trades']
        if len(trades) > 0:
            # first trade
            if self.trades is None:
                trade = trades[trades.ExitTime >= start].iloc[0]
                trade.EntryTime = start
                open_price = self.raw.loc[start].Open
                if trade.Size > 0:
                    trade.EntryPrice = open_price * (1 + COMMISSION)
                else:
                    trade.EntryPrice = open_price * (1 - COMMISSION)
                self.trades = pd.DataFrame([trade])

            elif any(trades.EntryTime == start):
                trade = trades[trades.EntryTime == start].iloc[0]
                size = self.trades.Size.iloc[-1]
                # signal change
                if np.sign(trade.Size) != np.sign(size):
                    open_price = self.raw.loc[start].Open
                    self.trades.ExitTime.iloc[-1] = start
                    self.trades.ExitPrice.iloc[-1] = open_price
                    if size > 0:    # long
                        return_pct = open_price
                        return_pct /= self.trades.EntryPrice.iloc[-1]
                    else:           # short
                        return_pct = self.trades.EntryPrice.iloc[-1]
                        return_pct /= open_price
                    return_pct -= 1.0
                    self.trades.ReturnPct.iloc[-1] = return_pct
                    self.trades = self.trades.append(trade)

        if self.t < len(self.data) - Candles.train - 2:
            self.t += 1
            self.pbar.update()
            self.obs = self.data[self.t:self.t + Candles.train + Candles.val]
            obs = self.preprocess(self.obs[:Candles.train].copy())
            reward = 0.0
            done = False
            info = {}
        else:
            self.pbar.close()
            obs = None
            reward = 0.0
            done = True
            info = {'trades': self.trades}

            # calculate total return
            info['return'] = np.prod(self.trades.ReturnPct + 1.0) - 1.0

            # calculate baselines
            start = self.data.index[-(Candles.train + Candles.test)]
            info['bnh'] = self.raw['Close'].iloc[-1] / self.raw['Open'][start] - 1.0
            bt = self.bt(data=self.raw.loc[start:])
            kwargs = {}
            for i in range(len(action)):
                key = 'n' + str(i + 1)
                kwargs[key] = getattr(bt._strategy, key + '_base')
            equity = bt.run(**kwargs)['_equity_curve']['Equity']
            info['baseline'] = equity.iloc[-1] / equity[start] - 1.0

        return obs, reward, done, info
