# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/30 16:56
import pandas as pd

df = pd.read_excel('data.xlsx')
# 假设上面有一个DataFrame叫做data
data = pd.DataFrame(df)
data2 = (data - data.min()) / (data.max() - data.min())  # 即简单实现标准化
print(data2)
data2.to_excel('01data.xlsx')