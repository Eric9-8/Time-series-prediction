# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/17 20:10
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
import codecs
from scipy import stats
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMAResults

# 读取
path = r'../原始数据/gx.csv'
df = pd.read_csv(path, nrows=600)
df.plot(figsize=(12, 8))
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# 判断1阶差分，或是2阶差分
diff1 = df.diff(1)

# 图像不好观察，继续做ADF单位根平稳性检验
print(sm.tsa.stattools.adfuller(df))
# diff1.plot(ax=ax1)
# plt.show()

# 看自相关图和偏相关图，设计模型
# fig1 = sm.graphics.tsa.plot_acf(df, lags=100)
# fig2 = sm.graphics.tsa.plot_pacf(df, lags=100)
# plt.show()
# 有5/1，5/0，0/5三种，通过判断aic，bic和hqic最小，选取51
arma_mod5_1 = sm.tsa.ARMA(df, (5, 1)).fit()
# arma_mod0_5 = sm.tsa.ARMA(df, (0, 1)).fit()


# print(arma_mod5_1.aic, arma_mod5_1.bic, arma_mod5_1.hqic)
# print(arma_mod0_5.aic, arma_mod0_5.bic, arma_mod0_5.hqic)


# 做DW分析，没有自相关性,1.59接近于2.
# print(sm.stats.durbin_watson(arma_mod5_1.resid.values))

resid = arma_mod5_1.resid  # 残差

# 观察残差是否符合正态分布
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

# Ljung-Box检验 当p-value<0.05（一般都用1%, 5%, 10%）, 拒绝原假设H0，结果显著，序列相关；当p-value>0.05，接受原假设H0，结果不显著，序列不相关，认为是白噪序列。
# r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))

# 预测
predict_sunspots = arma_mod5_1.predict(0, 600)
print(predict_sunspots)
x = pd.DataFrame(predict_sunspots)
x.to_csv('gx_pre.csv')

fig, ax = plt.subplots(figsize=(12, 8))
ax = df.plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

