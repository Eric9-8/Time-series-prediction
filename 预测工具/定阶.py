# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/17 21:04
import pandas as pd
import numpy as np
import seaborn as sns  # 热力图
import itertools
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller  # ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 画图定阶
from statsmodels.tsa.arima_model import ARIMA  # 模型
from statsmodels.tsa.arima_model import ARMA  # 模型
from statsmodels.stats.stattools import durbin_watson  # DW检验
from statsmodels.graphics.api import qqplot  # qq图

path = r'/Time-series/原始数据/pm.csv'
df = pd.read_csv(path, nrows=600)
df.plot(figsize=(12, 8))
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# 判断1阶差分，或是2阶差分
# diff1 = df.diff(1)
df = df.diff(1)
df[~(df == df)] = 0  # 补零操作
# 设置遍历循环的初始条件，以热力图的形式展示，跟AIC定阶作用一样
p_min = 0
q_min = 0
p_max = 5
q_max = 5
d_min = 0
d_max = 5
# 创建Dataframe,以BIC准则
results_aic = pd.DataFrame(index=['AR{}'.format(i)
                                  for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
# itertools.product 返回p,q中的元素的笛卡尔积的元组
for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1), range(q_min, q_max + 1)):
    if p == 0 and q == 0:
        results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(df, order=(p, d, q))
        results = model.fit()
        # 返回不同pq下的model的BIC值
        results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
    except:
        continue
results_aic = results_aic[results_aic.columns].astype(float)
# print(results_bic)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_aic,
                 # mask=results_aic.isnull(),
                 ax=ax,
                 annot=True,  # 将数字显示在热力图上
                 fmt='.2f',
                 )
ax.set_title('AIC')
plt.show()
train_results = sm.tsa.arma_order_select_ic(df, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)