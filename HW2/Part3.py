import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from LDA import LDAClassifier




# 創建一個二分類問題的數據集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練邏輯迴歸分類器
model = LDAClassifier()
model.fit(X_train, y_train)

# 使用分類器對測試集進行預測概率
y_scores = model.decision_function(X_test)[:, 1]  # 獲取正類別的預測概率

# 計算 ROC 曲線的 FPR, TPR
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 計算 AUC
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
