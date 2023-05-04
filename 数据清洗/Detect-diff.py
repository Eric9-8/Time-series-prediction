# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/29 20:57
from sklearn.svm import OneClassSVM
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

# py.init_notebook_mode(connected=True)

# 导入数据
data = np.loadtxt('6-9.csv', delimiter=',')

# 分配训练集和测试集
train_set = data[:3000, :]
test_set = data[-1000:, :]

# 异常检测
# 创建异常检测模型
one_svm = OneClassSVM(nu=0.1, kernel='rbf')
# 训练模型
one_svm.fit(train_set)
# 预测异常数据
pre_test_outliers = one_svm.predict(test_set)

# 异常结果统计
# 合并测试检测结果
total_test_data = np.hstack((test_set, pre_test_outliers.reshape(-1, 1)))

# 获取正常数据
normal_test_data = total_test_data[total_test_data[:, -1] == 1]
# 获取异常数据
outlier_test_data = total_test_data[total_test_data[:, -1] == -1]

# 输出异常数据结果
print('异常数据为：{}/{}'.format(len(outlier_test_data), len(total_test_data)))
# 异常数据为：9/100

# 可视化结果
py.plot([go.Scatter3d(x=total_test_data[:, 0], y=total_test_data[:, 1], z=total_test_data[:, 2],
                      mode='markers', marker=dict(color=total_test_data[:, -1], size=5))])
