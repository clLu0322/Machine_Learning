from IRIS_Split import iris_data_split
from qpsolvers import solve_qp
import numpy as np
import matplotlib.pyplot as plt

class SVMClassifier:
    def __init__(self, c=1, p_train=[], n_train=[]):
        self.c = c
        self.p_train_data = p_train
        self.n_train_data = n_train
        self.N = len(p_train) + len(n_train)

    def fit(self):
        self.y_train = []
        self.y_train += [1] * len(self.p_train_data)
        self.y_train += [-1] * len(self.n_train_data)
        self.X_train = self.p_train_data + self.n_train_data
        self.alpha_list = dual_problem(self.X_train, self.y_train, self.c, self.N)
        print(self.alpha_list)
        print(np.round(np.sum(self.alpha_list),4))

        self.sv_b = []
        for i in range(self.N):
            if 0 < self.alpha_list[i] < self.c:               
                b_i = 1/self.y_train[i]
                for j in range(self.N):
                    b_i -= self.alpha_list[j] * self.y_train[j] * np.dot(self.X_train[j].T, self.X_train[i])
                self.sv_b.append(float(b_i))
        print(self.sv_b)
        self.b = round(np.mean(self.sv_b), 4)
        print("bias:", self.b)

    def predict(self, p_label=1, n_label=-1, text_data=[]):
        predict_list = []
        
        for j in range(len(text_data)):
            decision = self.b
            for i in range(self.N):
                decision += self.alpha_list[i] * self.y_train[i]* np.dot(self.X_train[i].T, text_data[j])

            if decision >= 0:
                predict_list.append(p_label)
            else:
                predict_list.append(n_label)

        return predict_list


def dual_problem(X_train, y_train, c, N):
    y_train = np.array(y_train)
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = y_train[i] * y_train[j] * np.dot(X_train[i].T, X_train[j])
    q = -np.ones(N)
    A = y_train.reshape(1, -1).astype(np.float64)
    b = np.array([0.0])
    lb = np.zeros(N)
    ub = c * np.ones(N)
    alpha = solve_qp(P, q, G=None, h=None, A=A, b=b, lb=lb, ub=ub, solver='cvxopt')
    alpha = np.round(alpha, 4)
    return alpha

def plot_decision_boundary(svm, X_train, y_train, title="SVM Decision Boundary"):
    # 創建網格來評估模型
    x_min, x_max = min([x[0] for x in X_train]) - 1, max([x[0] for x in X_train]) + 1
    y_min, y_max = min([x[1] for x in X_train]) - 1, max([x[1] for x in X_train]) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # 預測網格上的每個點
    Z = np.array(svm.predict(text_data=grid))
    Z = Z.reshape(xx.shape)
    
    # 繪製決策邊界和支持向量
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
    
    support_vectors = [X_train[i] for i, alpha in enumerate(svm.alpha_list) if alpha > 1e-4]
    plt.scatter([x[0] for x in support_vectors], [x[1] for x in support_vectors], s=100, edgecolors='k', facecolors='none')

    plt.title(title)
    plt.xlabel('Feature 3')
    plt.ylabel('Feature 4')
    plt.show()

def main():
    np.set_printoptions(suppress=True, precision=4)
    data1, data2 = iris_data_split().feature_selection_and_split([2,3])
    test_data = data2[2] + data2[3]
    test_label = [2] * len(data2[2]) + [3] * len(data2[3])
    
    for c_value in [1, 10, 100]:
        print(f"\n\n\nC = {c_value}")
        SVM = SVMClassifier(c=c_value, p_train=data1[2], n_train=data1[3])
        SVM.fit()
        predict_result = SVM.predict(p_label=2, n_label=3, text_data=test_data)
        correct_count = 0
        for i in range(len(predict_result)):
            if predict_result[i] == test_label[i]:
                correct_count += 1
        CR = correct_count / len(test_data) * 100
        print(f"分類率: {CR:.2f}%")
        
        # 繪製決策邊界
        X_train = data1[2] + data1[3]
        y_train = [1] * len(data1[2]) + [-1] * len(data1[3])
        plot_decision_boundary(SVM, X_train, y_train, title=f"SVM Decision Boundary (C={c_value})")



    
if __name__ == "__main__":
    main()