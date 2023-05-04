# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/30 13:02
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('gx.csv')


def outliers(data, col_name, scale=3):  # scale--尺度 一般取3

    def box_plot_outliers(data_ser, box_scale):
        # IQR即尺度*（上四分位点-下四分位点）
        IQR = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - IQR  # 计算上边缘
        val_up = data_ser.quantile(0.75) + IQR  # 计算上下边缘
        rule_low = (data_ser < val_low)  # 小于下边缘的值
        rule_up = (data_ser > val_up)  # 大于上边缘的值
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]  # 对满足在rule_low以下或rule_up以上条件计数
    print("删除了: {} 个数据".format(len(index)))
    data_n.reset_index(drop=True, inplace=True)  # 取删除离群点后的数据
    print("剩余: {} 个数据".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]  # 计在下边缘以下的点
    print("小于下边缘线的数据详细:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]  # 计在上边缘以上的点
    outliers = data_series.iloc[index_up]
    print("大于上边缘线的数据详细:")
    print(pd.Series(outliers).describe())
    # 可视化数据
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, ax=ax[1])
    plt.show()
    return data_n


data = outliers(data, 'gx', scale=3)
data['gx'].replace(0, np.nan, inplace=True)
data['gx'].value_counts()
data.to_csv('gx_new.csv')
