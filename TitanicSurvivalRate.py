# Data from https://www.kaggle.com/c/titanic/data
#==============================================
from ast import increment_lineno
from cv2 import TermCriteria_COUNT
import pandas as pd
import numpy as np
# 繪圖套件
import matplotlib as plt

import matplotlib.gridspec as gridspec
import seaborn as sns
plt.style.use('ggplot')
#標籤編碼與熱鍵編碼
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#決策樹
from sklearn.tree import DecisionTreeClassifier
#隨機森林
from sklearn.ensemble import RandomForestClassifier

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

#匯入
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
submit = pd.read_csv('gender_submission.csv')

print(f'train {df_train.shape}')
display(df_train.head())

print(f'\ntest {df_test.shape}')
display(df_test.head())

def Col_Types(Data):
    Column_types = Data.dtypes.to_frame().reset_index()#判別每個欄位的型態
    Column_types.columns = ['ColumnName','Type']
    Column_types.sort_values(by = 'Type', inplace = True)
    return Column_types
display(Col_Types(df_train))

#補缺漏值
