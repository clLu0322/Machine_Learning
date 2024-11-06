import numpy as np
from IRIS_Split import iris_data_split
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class SVMClassifier:
    def __init__(self, c=10, p_train=[], n_train=[], poly=5):
        self.c = c
        self.p_train_data = p_train
        self.n_train_data = n_train
        self.N = len(p_train) + len(n_train)
        self.p = poly

    def fit(self):
        self.y_train = []
        self.y_train += [1] * len(self.p_train_data)
        self.y_train += [-1] * len(self.n_train_data)
        self.X_train = self.p_train_data + self.n_train_data
        self.alpha_list = dual_problem(self.X_train, self.y_train, self.c, self.N, self.p)
        print(self.alpha_list)
        print(np.round(np.sum(self.alpha_list), 4))

        self.sv_b = []
        for i in range(self.N):
            if 0 < self.alpha_list[i] < self.c:
                b_i = 1 / self.y_train[i]
                for j in range(self.N):
                    b_i -= self.alpha_list[j] * self.y_train[j] * polynomial_kernel(self.X_train[j], self.X_train[i], self.p)
                self.sv_b.append(float(b_i))
        print(self.sv_b)
        self.b = round(np.mean(self.sv_b), 4)
        print("bias:", self.b)

    def predict(self, p_label=1, n_label=-1, text_data=[]):
        predict_list = []

        for j in range(len(text_data)):
            decision = self.b
            for i in range(self.N):
                decision += self.alpha_list[i] * self.y_train[i] * polynomial_kernel(self.X_train[i], text_data[j], self.p)

            if decision >= 0:
                predict_list.append(p_label)
            else:
                predict_list.append(n_label)

        return predict_list


def dual_problem(X_train, y_train, c, N, poly):
    y_train = np.array(y_train)
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = y_train[i] * y_train[j] * polynomial_kernel(X_train[i], X_train[j], poly)
    q = -np.ones(N)
    A = y_train.reshape(1, -1).astype(np.float64)
    b = np.array([0.0])
    lb = np.zeros(N)
    ub = c * np.ones(N)
    alpha = solve_qp(P, q, G=None, h=None, A=A, b=b, lb=lb, ub=ub, solver='cvxopt')
    alpha = np.round(alpha, 4)
    return alpha


def polynomial_kernel(x, x_prime, p=1):
    dot_product = np.dot(np.array(x).flatten(), np.array(x_prime).flatten())
    return (dot_product) ** p


def plot_decision_boundary(svm_classifier, X, y, resolution=0.02):
    # 定義顏色圖，用於顯示正類、負類和邊界
    colors = ('lightblue', 'lightcoral', 'lightcyan')
    cmap = ListedColormap(colors)

    # 獲取邊界範圍
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 建立一個網格來進行決策
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # 預測每個網格點
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_classifier.predict(text_data=grid_points)
    Z = np.array(Z).reshape(xx.shape)

    # 畫出決策邊界
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 畫出所有的訓練資料點
    for idx, cl in enumerate(np.unique(y)):
        color = 'blue' if cl == 1 else 'red'
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.9, color=color, label=f'Class {cl}', edgecolor='k', s=50)
        
    # 圈出支持向量
    support_vectors = [sv for sv, alpha in zip(X, svm_classifier.alpha_list) if 0 < alpha < svm_classifier.c]
    support_vectors = np.array(support_vectors)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')


def main():
    data1, data2 = iris_data_split().feature_selection_and_split([2, 3])
    test_data = data2[2] + data2[3]
    test_label = [2] * len(data2[2]) + [3] * len(data2[3])

    y_train = np.array([1] * len(data1[2]) + [-1] * len(data1[3]))

    for poly_value in [1, 2, 3, 4, 5]:
        print(f"\n\n\npoly = {poly_value}")
        SVM = SVMClassifier(c=10, p_train=data1[2], n_train=data1[3], poly=poly_value)
        SVM.fit()
        predict_result = SVM.predict(p_label=2, n_label=3, text_data=test_data)

        correct_count = sum(1 for i in range(len(predict_result)) if predict_result[i] == test_label[i])
        CR = correct_count / len(test_data) * 100
        print(f"分類率: {CR:.2f}%")

        # 繪製分類邊界
        plt.figure(figsize=(15, 10), dpi=200)
        X_train_2d = np.array(data1[2] + data1[3])[:, :2]  # 只使用前兩個特徵進行可視化
        plot_decision_boundary(SVM, X_train_2d, y_train)
        plt.title(f'SVM Decision Boundary with Polynomial Kernel, Degree={poly_value}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
