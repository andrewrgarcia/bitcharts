from sklearn.svm import SVC
import pandas as pd
import numpy as np

def rollstats_MACD(df):
    """Compute the rolling statistics (also known as "financial indicators") using the Moving Averages Strategy (MACD) """
    MA1 = 12
    MA2 = 26

    df['fast MA'] = df['close'].rolling(window=MA1).mean()
    df['slow MA'] = df['close'].rolling(window=MA2).mean()

    df['rollstats'] = df['fast MA'] - df['slow MA']

    return df

def regimes(df):
    """Assign trading regimes to the data based on the computed rolling statistics."""

    rollstats = df['rollstats']

    crossover = 0

    df['regime'] = np.where(rollstats > crossover, 1, 0)
    df['regime'] = np.where(rollstats < -crossover, -1, df['regime'])

    df['signal'] = df['regime'].diff(
        periods=1)  # signal rolling difference
    df['regime_old'] = df['regime']

    return df

def strat_compute(df):
    """
    Compute the market and strategy returns based on the assigned trading regimes,
    and print the final return-on-investment as a message.

    The computed signal is used to fill the gaps in the signal of the trading regime.
    """

    df['market'] = np.log(df['close'] / df['close'].shift(1))

    df['strategy'] = df['regime'].shift(1) * df['market']

    df[['market', 'strategy']] = df[[
        'market', 'strategy']].cumsum().apply(np.exp)

    strategy_gain = df['strategy'].iloc[-1] - df['market'].iloc[-1]

    message = 'final return-on-investment: {:.2f}%'.format(
        strategy_gain*100)

    print(message)

    df["signal"][df["signal"] == 0.0] = np.nan

    return df


def strat_compute_mart(chart_obj):

  df = rollstats_MACD(chart_obj.dataframe)
  df = regimes(df)

  martingale(df) if mart else None

  df = strat_compute(df)


def strat_compute_SVC(chart_obj,lags=6):

  df = chart_obj.dataframe
  '''COMPUTE STRATEGY -- MARKET V STRAT RESULTS '''
  data = pd.DataFrame(df['close'])
  data['market'] = np.log(data / data.shift())
  data.dropna(inplace=True)

  '''LAGGED SIGNS OF LOG RATES OF RETURN 
  credit: Yves Hilpisch. 2018. Python for Finance: Analyze Big Financial Data (2nd. ed.). O'Reilly Media, Inc. '''
  # lags = 6
  cols = []
  for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = np.sign(data['market'].shift(lag))
    cols.append(col)
    data.dropna(inplace=True)

  model = SVC(gamma='auto')
  model.fit(data[cols], np.sign(data['market']))

  data['actual_sign'] = np.sign(data['market'])
  data['prediction'] = model.predict(data[cols])

  # **
  data['accurate'] = np.where(data['actual_sign']==data['prediction'],1,0)
  num_accurate = data['accurate'][data['accurate']>0].sum()
  accuracy = num_accurate / data['accurate'].shape[0]

  print(f"accuracy: {accuracy*100}%")

  cap_gains_tax = 0.28
  # cap_gains_tax = 0.0
  data['strategy'] = [ data['prediction'][i] * data['market'][i] * (1 - cap_gains_tax) if data['prediction'][i] < 0 else \
                      data['prediction'][i] * data['market'][i] * (1 - 0) for i in range(len(data['prediction'])) ]  # only taxes sale
  

  # data['strategy'] = data['prediction'] * data['market']
  data[['market','strategy']]=data[['market','strategy']].cumsum().apply(np.exp) 

  strategy_gain = data['strategy'].iloc[-1] - data['market'].iloc[-1]

  chart_obj.message = 'final anticipated ROI: {:.2f}%'.format(strategy_gain*100)
  print(chart_obj.message )

  return data 

def martingale(df):
  '''
  Implements the martingale betting strategy for trading cryptocurrencies.

  Parameters
  -------------
  df : pandas.DataFrame
      A DataFrame containing the trading data, including the 'signal' and 'regime' columns.

  Notes
  ---------
  This function uses the martingale betting strategy to determine position sizing for buying and selling cryptocurrencies.
  When a sell signal is triggered, the position size is increased by a factor of two if the profit from the trade is negative. 
  When a buy signal is triggered, the position size is reset to 1 if the last trade was a profitable sell. If the last trade was not profitable, the position size remains the same. 

  The 'regime' column of the DataFrame is multiplied by the position size to determine the size of each trade.
  '''

  # init variables
  memo = 1          #multiplies the regime col
  close_up, close_dn = 0,0
  PROFIT_SELL = 1
  MART_RESET = 0
  for i in range(df['regime'].shape[0]):

    if df['signal'][i] < 0:
      'sell'
      close_dn = df['close'][i]

      if memo < 1:
        memo *=-1
      
      PROFIT_SELL = close_dn - close_up

      if PROFIT_SELL < 0:
        memo *= -2
        MART_RESET = 0
      
      if PROFIT_SELL >= 0:
        memo = 1
        MART_RESET = 1


    if df['signal'][i] > 0:
      'buy'
      close_up = df['close'][i]

      if memo < 1:
        memo *=-1

      if MART_RESET:
        memo = 1
        MART_RESET = 0 

      'non-greedy [during sell] constraint'
      # PROFIT_BUY = -PROFIT_SELL
      # if PROFIT_BUY >= 0 and memo > 2:
      #   memo = -1


    df['regime'][i] *= memo
