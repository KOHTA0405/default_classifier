import pandas as pd
import numpy as np
import os

# csvファイルからtrain,testデータをそれぞれ読み込み
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# 目的変数を0 or 1に変換
df1['loan_status'] = df1['loan_status'].apply(lambda x : 1 if x=='ChargedOff' else 0)

# termを質的変数に変える
df1['term'].replace({'3 years':3, '5 years':5}, inplace=True)

# employment_lengthを質的変数に
df1['employment_length'].replace(['0 years', '1 year', '2 years', '3 years', '4 years', '5 years',
                                          '6 years', '7 years', '8 years', '9 years', '10 years'],
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

# 説明変数と目的変数に分ける
X = df1[['loan_amnt', 'term', 'interest_rate', 'employment_length', 'credit_score']]
y = df1['loan_status']

# ここまで処理をしたものを呼び出ししやすいファイルとして保存
np.savez_compressed('processed.npz', X=X, y=y)