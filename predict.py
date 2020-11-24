"""
# predict.py (c) 2019 matsuda
import cv2
import numpy as np
import joblib

# (1) データを整形する
file = input('画像ファイル名: ')
img = cv2.imread(file) # 新しいデータの読み込み
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) #データのリサイズ
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール化

a = np.array(g, dtype='float') # numpyの配列にする
a = (255 - a) / 255 # 反転（FMNISTの画像が白黒反転しているので）
a = a.flatten() # MLPClassifierの入力形式である、1次元配列に
        
# (2) 適用
mlp = joblib.load('model.pkl') # 学習済みのモデルの読み込み
r = mlp.predict([a]) # aがどのカテゴリかを計算する
print('分類結果:', r)
"""

import numpy as np
import pandas as pd
import joblib

def predict(params):
    model = joblib.load('nn.pkl')  # 学習済みのモデルの読み込み
    pred = model.predict(params)  # 未知データに対する予測値の取得
    return pred

# ['loan_amnt', 'term', 'interest_rate', 'employment_length', 'credit_score']
X_1 = int(input('借入総額をご入力ください：'))
X_2 = int(input('返済期間をご入力ください：'))
X_3 = float(input('金利をご入力ください：'))
X_4 = int(input('今の会社の勤続年数をご入力ください：'))
X_5 = float(input('信用スコアをご入力ください：'))
X_test = [[X_1, X_2, X_3, X_4, X_5]] # 特徴量のarray

print('=================================================')

if predict(X_test) == 0:
    print('ご入力内容に基づき融資を実行いたします')
else:
    print('申し訳ございませんが、融資の実行はいたしかねます')
