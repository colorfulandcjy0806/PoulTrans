import requests
from bs4 import BeautifulSoup
import csv
import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with 'Agg', 'Qt5Agg', etc., as needed
import matplotlib.pyplot as plt

#拿到页面源代码
#使用bs4进行解析，拿到数据

#写入地址，准备对数据进行爬取
url = "http://yss.mof.gov.cn/2021zyys/202103/t20210323_3674874.htm"
#写入请求标头
headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36 Edg/96.0.1054.43"
}

resp = requests.get(url)

#爬取后发现乱码
#从页面源代码中发现编码方式为utf-8，写入代码防止乱码
resp.encoding = 'utf-8'
resp.close()

#导入csv文件
f = open("finance.csv",
         mode="w",
         newline="",
         encoding="utf-8")
csvwriter = csv.writer(f)

#解析数据
#把页面源代码交给BeautifulSoup进行处理，生成bs对象
page = BeautifulSoup(resp.text,"html.parser")
#从bs对象中查找数据
#class为关键字，在其后添加关键字避免报错

table = page.find("table",class_="MsoNormalTable")
#print(table)
#找到所有的行
#对数据进行切片处理
trs = table.find_all("tr")[2:]

for tr in trs:
    #拿到每行中的td
    tds = tr.find_all("td")
    #拿到被标签标记的内容
    name = tds[0].text
    num_20 = tds[1].text
    num_20_1 = tds[2].text
    num_21 = tds[3].text
    csvwriter.writerow([name,num_20,num_20_1,num_21])

#关闭文件
f.close()

import pandas as pd
import numpy as np

file_path = "finance.csv"
df = pd.read_csv(file_path,header=None)

#写入列名
df.columns = ['地区',
              '2020执行数',
              '2020执行数(剔除特殊转移支付)',
              '2021预算数']
df.duplicated()

from matplotlib import pyplot as plt
import matplotlib

#设置中文显示，解决乱码问题
font = {'family' : 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '12'}
matplotlib.rc("font",**font)
matplotlib.rc("font",
              family='MicroSoft YaHei',
              weight="bold")

# ==========
#设置中文显示，解决乱码问题
font = {'family' : 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '12'}
matplotlib.rc("font",**font)
matplotlib.rc("font",
              family='MicroSoft YaHei',
              weight="bold")
fig, ax = plt.subplots(figsize=(20, 8))
ax.stem(df['地区'],
         df['2021预算数'])

#设置标签
plt.xlabel("地区")
plt.ylabel("人民币       单位（亿元）")
plt.title("2021年中央对地方一般公共预算转移支付分地区情况汇总表")
#X轴翻转90度
plt.xticks(rotation=90)
#打印图片
plt.show()
#============