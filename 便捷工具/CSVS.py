# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/3/2 11:53
import pandas as pd


def xlsx_to_csv_pd():
    data_xls = pd.read_excel('d.xlsx', index_col=0, sheet_name=0)
    data_xls.to_csv('gx.csv', encoding='utf-8')


if __name__ == '__main__':
    xlsx_to_csv_pd()
