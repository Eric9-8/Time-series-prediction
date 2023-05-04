# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/30 16:01
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('gx.csv')

data2 = savgol_filter(data, 203, 3, mode='nearest')
plt.plot(data)
plt.plot(data2)
plt.show()
