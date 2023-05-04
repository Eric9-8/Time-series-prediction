# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/22 9:47
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = r'../原始数据/gz.csv'
path1 = r'原始数据/pm.csv'
df = pd.read_csv(path, nrows=600)
df1 = pd.read_csv(path1, nrows=600)
df.plot(figsize=(12, 8))
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# 判断1阶差分，或是2阶差分
diff1 = df.diff(1)
diff2 = df1.diff(1)
# 图像不好观察，继续做ADF单位根平稳性检验
print(sm.tsa.stattools.adfuller(df))
diff1.plot(ax=ax1)
# plt.show()
