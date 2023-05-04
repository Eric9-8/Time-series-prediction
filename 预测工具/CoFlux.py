# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/9 10:10
import numpy as np
import math
from sklearn import preprocessing


# 特征提取，做导数和小波分析

def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total) / lenth
    tempsum = sum([pow(data[i] - ave, 2) for i in range(lenth)])
    tempsum = pow(float(tempsum) / lenth, 0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave) / tempsum
    return data


path1 = 'pm特征.csv'
g = np.loadtxt(path1, delimiter=',')
G = (g - min(g)) / (max(g) - min(g))  # 归一化处理
# G = Z_Score(g)  # 归一化处理
path2 = 'gy特征.csv'
h = np.loadtxt(path2, delimiter=',')
H = (h - min(h)) / (max(h) - min(h))  # 归一化处理
# H = Z_Score(h)

Rgh = []
# 放大器'
# G = amplify(G)
# H = amplify(H)

for s in range(-600, 600):
    Zr = np.zeros((abs(s)), dtype=int)  # 0数组
    if s < 0:
        s = -s
        Gt = G
        # 删除前s个数据, 0代表行
        for i in range(1, s + 1):
            Gt = np.delete(Gt, 0)
        Gs = np.concatenate((Gt, Zr))  # 合并数
    else:
        Gt = G
        # 删除后s个数据, 0代表行
        for i in range(1, s + 1):
            Gt = np.delete(Gt, -1)
        Gs = np.concatenate((Zr, Gt))  # 合并数组
    rgh = np.dot(Gs, H)
    Rgh.append(rgh)
Rgg = np.dot(G, G)
# print(Rgg)
Rhh = np.dot(H, H)
# print(Rhh)
# print(N)
CC = []
for rgh in Rgh:
    cc = rgh / (math.sqrt(Rgg * Rhh))
    CC.append(cc)
minCC = min(CC)
print(minCC)
maxCC = max(CC)
print('==============')
print(maxCC)
s1 = np.argmin(CC)
s2 = np.argmax(CC)
print(s1)
print(s2)
