# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/30 17:35
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from pyhht import EMD
from scipy.signal import hilbert
import tftb.processing
import pandas as pd


# 定义HHT的计算分析函数
def HHTAnalysis(eegRaw, fs):
    # 进行EMD分解
    decomposer = EMD(eegRaw)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 分解后的组分数
    n_components = imfs.shape[0]
    # 定义绘图，包括原始数据以及各组分数据
    fig, axes = plt.subplots(n_components + 1, 2, figsize=(10, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0][0].plot(eegRaw)
    # 原始数据的Hilbert变换
    eegRawHT = hilbert(eegRaw)
    # 绘制原始数据Hilbert变换的结果
    axes[0][0].plot(abs(eegRawHT))
    # 设置绘图标题
    axes[0][0].set_title('Raw Data')
    # 计算Hilbert变换后的瞬时频率
    instf, timestamps = tftb.processing.inst_freq(eegRawHT)
    # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
    axes[0][1].plot(timestamps, instf * fs)
    # 计算瞬时频率的均值和中位数
    axes[0][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))

    # 计算并绘制各个组分
    for iter in range(n_components):
        # 绘制分解后的IMF组分
        axes[iter + 1][0].plot(imfs[iter])
        # 计算各组分的Hilbert变换
        imfsHT = hilbert(imfs[iter])
        # 绘制各组分的Hilber变换
        axes[iter + 1][0].plot(abs(imfsHT))
        # 设置图名
        axes[iter + 1][0].set_title('IMF{}'.format(iter))
        # 计算各组分Hilbert变换后的瞬时频率
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
        axes[iter + 1][1].plot(timestamps, instf * fs)
        # 计算瞬时频率的均值和中位数
        axes[iter + 1][1].set_title(
            'Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
    plt.show()


# 定义HHT的滤波函数，提取部分EMD组分
def HHTFilter(eegRaw, componentsRetain):
    # 进行EMD分解
    decomposer = EMD(eegRaw)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 选取需要保留的EMD组分，并且将其合成信号
    eegRetain = np.sum(imfs[componentsRetain], axis=0)

    # 绘图
    plt.figure(figsize=(10, 7))
    # 绘制原始数据
    plt.plot(eegRaw, label='RawData')
    # 绘制保留组分合成的数据
    plt.plot(eegRetain, label='HHTData')
    # 绘制标题
    plt.title('RawData-----HHTData')
    # 绘制图例
    plt.legend()
    plt.show()
    return eegRetain


if __name__ == '__main__':
    # 示例数据分析
    # 生成0-1时间序列，共100个点
    # t = np.linspace(0, 1, 100)
    # # 生成频率为5Hz、10Hz、25Hz的正弦信号累加
    # modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 25 * t)
    # # 信号和时间累加，相当于添加噪声
    # eegRaw = modes + t
    # # fs为采样频率，用于正则化频率和真实频率的转换
    # fs = 100
    # # 进行HHT分析
    # HHTAnalysis(eegRaw, fs)
    # # 选取需要保留的信号进行合成，也就是相当于滤波
    # eegRetain = HHTFilter(eegRaw, [0, 1, 2, 3])
    # 真实数据分析
    data = pd.read_excel('gz.xlsx')
    data = np.array(data)
    freq = 300
    HHTAnalysis(data, freq)
    HHTFilter(data, [0, 1, 2, 3, 4, 5, 6, 7])
    # # 加载fif格式的数据
    # epochs = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')
    # # 获取采样频率
    # sfreq = epochs.info['sfreq']
    # # 想要分析的数据
    # eegData = epochs.get_data()[0][0]
    # HHTAnalysis(eegData, sfreq)
    # HHTFilter(eegData, [0, 1, 2, 3, 4, 5, 6, 7, 8])
