# -*- coding: utf-8 -*-
# @Time : 2022/8/16 9:40
# @Author : yysgz
# @File : test.py
# @Project : tri-party deep network representation.ipynb

import tkinter as tk
import random
import threading
import time

import  pickle
import pandas as pd

data = pd.DataFrame()
# 写入数据
pkl_file = open('D:/raw_data', 'wb')
pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)
pkl_file.close()

# 读入数据
pkl_file_rb = open('D/raw_data', 'rb')
new_data = pickle.load(pkl_file_rb)