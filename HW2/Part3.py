from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from LDA import LDAClassifier, iris_data_split


def main():
    data1_dict, data2_dict = iris_data_split().feature_selection_and_split([0,1])
    test_data = data2_dict[2] + data2_dict[3]
    y_test = [2] * len(data2_dict[2]) + [3] * len(data2_dict[3])
    # 訓練邏輯迴歸分類器
    model = LDAClassifier(p_label = 3, n_label = 2, cp=1, cn=1, p_train=data1_dict[3], n_train = data1_dict[2])
    model.fit()
    y_scores = model.get_scores(test_data)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores,pos_label=3)# 計算 ROC 曲線的 FPR, TPR
    roc_auc = auc(fpr, tpr)# 計算 AUC

    # 繪製 ROC 曲線
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
