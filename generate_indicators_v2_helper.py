import time
import numpy as np
import pandas as pd
from ta.utils import dropna
from ta.trend import *
from ta.momentum import *
from ta.volatility import *
from ta.volume import *
import plotly.express as px
import quantstats as qs

def generate_indicators(df):

    # OHLC_SHIFTS
    open_data = df["open"].shift(1)
    high_data = df["high"].shift(1)
    low_data = df["low"].shift(1)
    close_data = df["close"].shift(1)
    volume_data = df["volume"].shift(1)

    # ACCURACY REFERENCE
    df['up'] = df['close'] >= df['open']

    # TREND
    AROON = AroonIndicator(close=close_data, window=50)
    CCI = CCIIndicator(high=high_data, low=low_data, close=close_data, window=35)
    DPOI = DPOIndicator(close=close_data, window=50)
    EMA_1 = EMAIndicator(close=close_data, window=35)
    EMA_2 = EMAIndicator(close=close_data, window=77)
    EMA_3 = EMAIndicator(close=close_data, window=133)
    EMA_4 = EMAIndicator(close=close_data, window=224)
    df['dpoi'] = DPOI.dpo()
    df['aroon'] = AROON.aroon_indicator()
    df['cci'] = CCI.cci()
    df['ema_1'] = EMA_1.ema_indicator()
    df['ema_2'] = EMA_2.ema_indicator()
    df['ema_3'] = EMA_3.ema_indicator()
    df['ema_4'] = EMA_4.ema_indicator()
    df['ema_1_diff'] = (df['ema_1'] - df['open']) / df['open']
    df['ema_2_diff'] = (df['ema_2'] - df['open']) / df['open']
    df['ema_3_diff'] = (df['ema_3'] - df['open']) / df['open']
    df['ema_4_diff'] = (df['ema_4'] - df['open']) / df['open']

    # VOLUME
    ADI = AccDistIndexIndicator(high=high_data, low=low_data, close=close_data, volume=volume_data)
    CHAIKIN = ChaikinMoneyFlowIndicator(high=high_data, low=low_data, close=close_data, volume=volume_data, window=32)
    EOMV = EaseOfMovementIndicator(high=high_data, low=low_data, volume=volume_data, window=24)
    FORCE = ForceIndexIndicator(close=close_data, volume=volume_data, window=15)
    MFI = MFIIndicator(high=high_data, low=low_data, close=close_data, volume=volume_data, window=24)
    NVI = NegativeVolumeIndexIndicator(close=close_data, volume=volume_data)
    VPT = VolumePriceTrendIndicator(close=close_data, volume=volume_data)
    VWAP = VolumeWeightedAveragePrice(high=high_data, low=low_data, close=close_data, volume=volume_data, window=24)

    df['adi'] = ADI.acc_dist_index()
    df['chaikin'] = CHAIKIN.chaikin_money_flow()
    df['eomv'] = EOMV.ease_of_movement()
    df['force'] = FORCE.force_index()
    df['mfi'] = MFI.money_flow_index()
    df['nvi'] = NVI.negative_volume_index()
    df['vpt'] = VPT.volume_price_trend()
    df['vwap'] = VWAP.volume_weighted_average_price()

    # VOLATILITY
    ATR = AverageTrueRange(high=high_data, low=low_data, close=close_data, window=18)

    df['atr'] = ATR.average_true_range()

    # MOMENTUM
    AWESOME = AwesomeOscillatorIndicator(high=high_data, low=low_data, window1=7, window2=35)
    KAUFMAN = KAMAIndicator(close=close_data, window=14)
    PPO = PercentagePriceOscillator(close=close_data)
    PVO = PercentageVolumeOscillator(volume=volume_data)
    ROC = ROCIndicator(close=close_data, window=20)
    RSI = RSIIndicator(close=close_data, window=7)
    STOCH_RSI = StochRSIIndicator(close=close_data, window=24)
    STOCH = StochasticOscillator(high=high_data, low=low_data, close=close_data)
    TSI = TSIIndicator(close=close_data)
    ULT = UltimateOscillator(high=high_data, low=low_data, close=close_data)
    WILLIAMS = WilliamsRIndicator(high=high_data, low=low_data, close=close_data)

    df['ao'] = AWESOME.awesome_oscillator()
    df['ao_diff'] = (df['ao'] - df['open']) / df['open']

    df['kaufman'] = KAUFMAN.kama()
    df['kaufman_diff'] = (df['kaufman'] - df['open']) / df['open']

    df['rsi'] = RSI.rsi()
    df['rsi_stoch_k'] = STOCH_RSI.stochrsi_k()
    df['rsi_stoch_d'] = STOCH_RSI.stochrsi_d()

    df['ppo'] = PPO.ppo_signal()
    df['pvo'] = PVO.pvo_signal()
    df['roc'] = ROC.roc()
    df['stoch'] = STOCH.stoch()
    df['tsi'] = TSI.tsi()
    df['ult'] = ULT.ultimate_oscillator()
    df['williams'] = WILLIAMS.williams_r()

    return df

def generate_chart(df_chart, predictions, coin, factors):

    df = pd.DataFrame()

    df['time'] = df_chart['time']
    df['dir'] = predictions
    df['delta'] = (df_chart['close'] - df_chart['open']) / df_chart['open']

    df['dir'] = df['dir'].replace({True: 1, False: -1})
    df['fee'] = abs(df['dir'].diff() * .001)

    df['real_delta_test'] = df['delta'] * df['dir'].astype(int) - df['fee']

    df = df.dropna()

    df['cumsum_strat'] = np.cumprod(1 + df['real_delta_test'].values) - 1
    df['cumsum_coin'] = np.cumprod(1 + df['delta'].values) - 1

    fig = px.line(df, x="time", y=["cumsum_strat", "cumsum_coin"], title='Cumulative Returns: ' + coin + ' ... ' + str(factors))
    fig.show()

def generate_returns(df_chart, predictions):

    df = pd.DataFrame()

    df['time'] = df_chart['time']
    df['dir'] = predictions
    df['delta'] = (df_chart['close'] - df_chart['open']) / df_chart['open']

    df['dir'] = df['dir'].replace({True: 1, False: -1})
    df['fee'] = abs(df['dir'].diff() * .001)

    df['real_delta_test'] = df['delta'] * df['dir'].astype(int) - df['fee']

    df = df.dropna()

    df['cumsum_strat'] = np.cumprod(1 + df['real_delta_test'].values) - 1
    df['cumsum_coin'] = np.cumprod(1 + df['delta'].values) - 1

    sharpe = qs.stats.sharpe(df['real_delta_test'])
    sortino = qs.stats.sortino(df['real_delta_test'])

    return [df['cumsum_coin'].iloc[-1], df['cumsum_strat'].iloc[-1], sharpe, sortino]