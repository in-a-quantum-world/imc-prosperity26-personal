
import pandas as pd, numpy as np, glob, os, textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

price_files=sorted(glob.glob('/mnt/data/prices_round_1_day_*.csv'))
trade_files=sorted(glob.glob('/mnt/data/trades_round_1_day_*.csv'))
prices=pd.concat([pd.read_csv(f, sep=';') for f in price_files], ignore_index=True)
trade_list=[]
for f in trade_files:
    df=pd.read_csv(f, sep=';')
    day=int(os.path.basename(f).split('day_')[1].split('.csv')[0])
    df['day']=day
    trade_list.append(df)
trades=pd.concat(trade_list, ignore_index=True)

prices=prices.dropna(subset=['bid_price_1','ask_price_1']).copy()
prices['t_global']=(prices['day']-prices['day'].min())*1_000_000+prices['timestamp']
trades['t_global']=(trades['day']-prices['day'].min())*1_000_000+trades['timestamp']
prices['spread']=prices['ask_price_1']-prices['bid_price_1']
prices['imbalance1']=(prices['bid_volume_1'].abs()-prices['ask_volume_1'].abs())/(prices['bid_volume_1'].abs()+prices['ask_volume_1'].abs())
prices['microprice']=(prices['ask_price_1']*prices['bid_volume_1'].abs()+prices['bid_price_1']*prices['ask_volume_1'].abs())/(prices['bid_volume_1'].abs()+prices['ask_volume_1'].abs())

summary_rows=[]
for (prod,day), sub in prices.groupby(['product','day']):
    sub=sub.sort_values('timestamp')
    x=sub['timestamp'].values
    y=sub['mid_price'].values
    coef=np.polyfit(x,y,1)
    yhat=np.polyval(coef,x)
    ss_res=((y-yhat)**2).sum()
    ss_tot=((y-y.mean())**2).sum()
    r2=1-ss_res/ss_tot
    sub2=sub.copy()
    sub2['next_ret']=sub2['mid_price'].shift(-1)-sub2['mid_price']
    summary_rows.append({
        'product': prod,
        'day': day,
        'mean_mid': sub['mid_price'].mean(),
        'mid_std': sub['mid_price'].std(),
        'mean_spread': sub['spread'].mean(),
        'trend_slope_per_100_ts': coef[0]*100,
        'trend_r2': r2,
        'imbalance_corr_next_ret': sub2['imbalance1'].corr(sub2['next_ret']),
        'trade_count': int((trades['symbol']==prod).sum() if 'symbol' in trades.columns else 0)
    })
summary_df=pd.DataFrame(summary_rows)
summary_df.to_csv('/mnt/data/prosperity4_round1_summary_stats.csv', index=False)
print(summary_df)
