# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/30 13:31
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LOF():
    def __init__(self, data, k, a=2):
        self.data = data
        self.k = k
        self.a = a

    def adaption(self):
        '''
        先判定k值是否小于数据长度，再判定数据类型是否符合
        '''
        # import numpy as np
        arr = self.data
        if self.k > len(self.data):
            raise KeyError('The value of K is larger than the length of array')
        if 'DataFrame' in str(type(self.data)):
            try:
                import pandas as pd
                arr = self.data.values
            except:
                raise ModuleNotFoundError('Either change your data type to numpy.ndarray or install Pandas')
        elif type(self.data) is list:
            arr = np.array(self.data)
        return arr

    def dist_table(self):
        '''
        利用numpy的向量范数计算各点间的距离，a系数及ord=1即曼哈顿距离，=2即欧几里得距离，np.inf(正无穷)即闵可夫斯基距离
        '''
        # import numpy as np
        arr = self.adaption()
        dc = np.zeros(shape=(arr.shape[0], arr.shape[0]))
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr)):
                dis = np.linalg.norm(arr[i, :] - arr[j, :], ord=self.a)
                temp.append(dis)
            temp = np.array(temp)
            dc[:, i] = temp
        return dc

    def kdist(self):
        '''
        The distance between o and the k-th nearest neighbor
        '''
        # import numpy as np
        dt = self.dist_table()
        kd = []
        nn = {}  # k个最邻点index
        for i in range(len(dt)):
            dl = dt[:, i].tolist()
            mindist = sorted(dl, reverse=False)[1:self.k + 1]
            kd.append(max(mindist))
            index = []
            for m in mindist:
                idx = [y for y, x in enumerate(dl) if x == m]  # 找该距离对应值的索引
                for j in idx:
                    if j not in index and j != i:  # 不添加重复的index,且不添加与自己相同的索引
                        index.append(j)
                        break
            nn[i] = index
        return kd, nn, dt

    def rbdist(self):
        '''
        设Reach_distk(i, j)，i为行，j为列
        Reach_distk(i, j) = max{dist(i, j), k-distance(j)}
        '''
        # import numpy as np
        kd, nn, dt = self.kdist()
        rd = np.zeros(shape=(dt.shape[0], dt.shape[1]))
        for i in range(len(rd)):
            for j in nn[i]:
                rd[i, j] = max([dt[i, j]] + [kd[j]])
        return rd

    def lrd(self):
        '''
        Local Reachability Density
        Nk(P) / ∑Nk(P)Reach_distk(i, j)
        '''
        # import numpy as np
        rd = self.rbdist()
        lrd_ls = []
        for i in range(len(rd)):
            rs = rd[i].sum()
            if rs == 0:  # 设有k+1个以上的点在同一个坐标，则他们的rbdist=0，这种情况令其=k/0.1
                lrd_ls.append(self.k / 0.1)
            else:
                lrd_ls.append(self.k / rs)
        return lrd_ls

    def lof(self):
        '''
        ∑Nk(P)(lrd(o)/lrd(p)) / |Nk(P)|
        '''
        # import numpy as np
        ld = self.lrd()
        _, nn, _ = self.kdist()
        lof_ls = []
        for i in range(len(ld)):
            sum_lrd_o = 0
            for j in range(self.k):
                sum_lrd_o += ld[nn[i][j]]
            if sum_lrd_o == 0 and ld[i] == 0:  # 设两者都为0(不存在只有ld[i]=0的情况)时，除法等于0.1/k
                lof_ls.append(0.1 / self.k)
            else:
                lof_ls.append((sum_lrd_o / ld[i]) / self.k)
        return np.array(lof_ls).reshape(-1, 1)

    def pred(self):
        '''
        Boxplot method to detect the 1-dimentional LOF outliers
        '''
        lof_result = self.lof()
        q1 = np.quantile(lof_result, 0.25, interpolation='lower')
        q3 = np.quantile(lof_result, 0.75, interpolation='higher')
        iqr = q3 - q1
        up_outliers = 1.5 * iqr + q3
        lof_result = np.where(lof_result > up_outliers, -1, 1)
        return lof_result


# np.random.seed(42)
#
# # Generate train data
# X_inliers = 0.3 * np.random.randn(100, 2)
# X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
#
# # Generate some outliers
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# X = np.r_[X_inliers, X_outliers]
# n_outliers = len(X_outliers)
# ground_truth = np.ones(len(X), dtype=int)
# ground_truth[-n_outliers:] = -1
X = pd.read_csv('../6-9.csv', index_col='Pm')
X = np.array(X)
# 计算lof
lof = LOF(X, 20).lof()
y_pred = LOF(X, 20).pred()

# 绘图
pred_normal = X[np.where(y_pred == 1)[0]]
pred_outliers = X[np.where(y_pred == -1)[0]]
lof_outliers = lof[np.where(y_pred == -1)[0]]

# 数据分布原图
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title('Original Distribution', fontsize=20)
plt.show()

# lof算法处理后
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(pred_normal[:, 0], pred_normal[:, 1], marker='.', label='Normal Points')
ax.scatter(pred_outliers[:, 0], pred_outliers[:, 1], marker='.', label='Outliers')
ax.legend()
ax.set_title('Local Outlier Factor', fontsize=20)
for i in range(len(lof_outliers)):
    plt.text(pred_outliers[i, 0], pred_outliers[i, 1], round(lof_outliers[i][0], 2), ha='center', va='bottom',
             fontsize=10)
plt.show()