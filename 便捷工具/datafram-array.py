# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/15 8:48
import pandas
from pandas import read_excel
from sklearn import preprocessing

dataset = read_excel('D:/data/test.xlsx', header=0, index_col=0)
values = dataset.values  # dataframe转换为array
values = values.astype('float32')  # 定义数据类型

data = preprocessing.scale(values)
df = pandas.DataFrame(data)  # 将array还原为dataframe

df.columns = dataset.columns  # 命名标题行

df.to_excel('D:/data/result.xlsx', index=None)  # 另存为excel，删除索引
