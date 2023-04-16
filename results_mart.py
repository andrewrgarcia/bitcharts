import methods
# import bitcharts as bits 
import cointables as coin
import mplfinance as mpf

from binance.client import Client
import config

def backtest(mart,title_extra=''):

  chart = coin.Chart(client = Client(config.API_KEY, config.API_SECRET))

  chart.coinGET()
  
  df = methods.rollstats_MACD(chart.dataframe)
  df = methods.regimes(df)

  methods.martingale(df) if mart else None

  df = methods.strat_compute(df)

  'PLOTTING'
  ap = [
        mpf.make_addplot(df['market'],panel=1,type='line',ylabel='strategy',secondary_y=False),
        mpf.make_addplot(df['strategy'],panel=1,type='line',secondary_y=False),
        mpf.make_addplot(df['signal'],panel=2,type='scatter',\
                         markersize=100,marker='v',ylabel='signal'),
        mpf.make_addplot(df['regime_old'],panel=2,type='line',ylabel='regime'),
        mpf.make_addplot(df['regime'],panel=2,type='line',ylabel='regime'),
  ]

  s  = mpf.make_mpf_style(base_mpf_style='nightclouds',figcolor='#222')

  mpf.plot(df, title=chart.coin+chart.market+" ({} candles) {}".format(chart.candles,title_extra),\
          mav=(12,26), type='candle',ylabel='Candle',addplot=ap,panel_ratios=(2,1),xlabel=chart.message,\
          figratio=(1,1),figscale=1.5,style=s)
  
string='Without Martingale'
print('\n'+string)
backtest(False,'--- '+string)

string='Martingale-Hedged strategy'
print('\n'+string)
backtest(True,'--- '+string)
