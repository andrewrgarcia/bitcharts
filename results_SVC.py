import cointables as coin
import methods

from binance.client import Client
import config

import mplfinance as mpf

def backtest():
  
  chart = coin.Chart(client = Client(config.API_KEY, config.API_SECRET))

  chart.coin = "ETH"
  chart.market = "USDT"
  chart.candles = "30m"
  
  chart.coinGET(num_candles=500)
  
  df = chart.dataframe

  data = methods.strat_compute_SVC(chart)

  diff_rows = df['close'].shape[0] - data['strategy'].shape[0]

  'PLOTTING'
  ap = [
        mpf.make_addplot(data['market'],panel=1,type='line',ylabel='strategy',secondary_y=False),
        mpf.make_addplot(data['strategy'],panel=1,type='line',secondary_y=False),
        mpf.make_addplot(data['actual_sign'],panel=2,type='line',ylabel='prediction'),
        mpf.make_addplot(data['prediction'],panel=2,type='line',ylabel='prediction')
  ]

  s  = mpf.make_mpf_style(base_mpf_style='nightclouds',figcolor='#222')


  mpf.plot(df[diff_rows:], title=chart.coin+chart.market+"({} candles) -- Support Vector Machines".format(chart.candles),\
          mav=(12,26),type='candle',ylabel='Candle',addplot=ap,panel_ratios=(2,1),xlabel=chart.message,\
          figratio=(1,1),figscale=1.5,style=s)
  

backtest()