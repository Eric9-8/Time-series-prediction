# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/18 12:44
import math
import numpy as np


def amplify(data):
    x = []
    for val in data:
        if val < 0:
            val = abs(val)
            val = val * 10
            x.append(val)
            data = min(x) * 0.5
            data = -(math.exp(data)) + 1
        else:
            data = val * 10
            x.append(val)
            data = min(x) * 0.5
            data = math.exp(data) - 1
    print(data)
    return data
