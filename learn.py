from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import joblib

# 前処理済みのデータを読み込み
data = np.load('processed.npz')
X = data['X']
y = data['y']
X_train, X_val, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

# ニューラルモデルの作成
model = MLPClassifier(solver="sgd", random_state=0, max_iter=100)

# 学習実施
model.fit(X_train, y_train)

# 学習済みモデルの保存
joblib.dump(model, 'nn.pkl', compress=True)