from IRIS_Split import iris_data_split
from qpsolvers import solve_qp
import numpy as np
import pandas as pd
import os

class SVMClassifier:
    def __init__(self, c=1, p_train=[], n_train=[], sigma=1):
        self.c = c
        self.p_train_data = p_train
        self.n_train_data = n_train
        self.N = len(p_train) + len(n_train)
        self.sigma = sigma

    def fit(self):
        self.y_train = []
        self.y_train += [1] * len(self.p_train_data)
        self.y_train += [-1] * len(self.n_train_data)
        self.X_train = self.p_train_data + self.n_train_data
        self.alpha_list = dual_problem(self.X_train, self.y_train, self.c, self.N, self.sigma)
        #print(self.alpha_list)
        #print(np.round(np.sum(self.alpha_list),4))

        self.sv_b = []
        for i in range(self.N):
            if 0 < self.alpha_list[i] < self.c:               
                b_i = 1/self.y_train[i]
                for j in range(self.N):
                    b_i -= self.alpha_list[j] * self.y_train[j] * rbf_kernel(self.X_train[j], self.X_train[i],self.sigma)
                self.sv_b.append(float(b_i))
        #print(self.sv_b)

        self.b = round(np.mean(self.sv_b), 4)
        #print("bias:", self.b)

    def predict(self, p_label=1, n_label=-1, text_data=[]):
        predict_list = []
        
        for j in range(len(text_data)):
            decision = self.b
            for i in range(self.N):
                decision += self.alpha_list[i] * self.y_train[i]* rbf_kernel(self.X_train[i], text_data[j],self.sigma)

            if decision >= 0:
                predict_list.append(p_label)
            else:
                predict_list.append(n_label)

        return predict_list
    

def dual_problem(X_train, y_train, c, N, sigma):
    y_train = np.array(y_train)
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = y_train[i] * y_train[j] * rbf_kernel(X_train[i], X_train[j], sigma)
    q = -np.ones(N)
    A = y_train.reshape(1, -1).astype(np.float64)
    b = np.array([0.0])
    lb = np.zeros(N)
    ub = c * np.ones(N)
    alpha = solve_qp(P, q, G=None, h=None, A=A, b=b, lb=lb, ub=ub, solver='cvxopt')
    alpha = np.round(alpha, 4)
    return alpha


def rbf_kernel(x, x_prime, sigma):
    x = np.array(x)
    x_prime = np.array(x_prime)
    distance_squared = np.linalg.norm(x - x_prime) ** 2
    return np.exp(-distance_squared / (2 * sigma**2))


def main():
    np.set_printoptions(suppress=True, precision=4)
    data1, data2 = iris_data_split().feature_selection_and_split([0, 1, 2, 3])
    X_train12 = np.array(data1[1] + data1[2])
    X_train13 = np.array(data1[1] + data1[3])
    X_train23 = np.array(data1[2] + data1[3])
    X2_train12 = np.array(data2[1] + data2[2])
    X2_train13 = np.array(data2[1] + data2[3])
    X2_train23 = np.array(data2[2] + data2[3]) 
    y_train = np.array([1] * len(data1[1]) + [-1] * len(data1[2]))

    test_data = np.array(data2[1] + data2[2] + data2[3])
    test_data2 = np.array(data1[1] + data1[2] + data1[3])
    test_label = np.array([1] * len(data2[1]) + [2] * len(data2[2]) + [3] * len(data2[3]))

    c_values = [1, 5, 10, 50, 100, 500, 1000]
    sigma_logs = [i for i in range(-100, 105, 5)]
    table_data = []    


    for sigma_log in sigma_logs:
        sigma_value = 1.05 ** sigma_log
        row = []
        print(f"sigma (log_scale): {sigma_log}")
        for c_value in c_values:
            print(f"C = {c_value}:  ", end='')
            SVM12 = SVMClassifier(c=c_value, p_train=X_train12[y_train == 1].tolist(), n_train=X_train12[y_train == -1].tolist(), sigma=sigma_value)
            SVM12.fit()
            predict_result12 = SVM12.predict(p_label=1, n_label=2, text_data=test_data)

            SVM13 = SVMClassifier(c=c_value, p_train=X_train13[y_train == 1].tolist(), n_train=X_train13[y_train == -1].tolist(), sigma=sigma_value)
            SVM13.fit()
            predict_result13 = SVM13.predict(p_label=1, n_label=3, text_data=test_data)

            SVM23 = SVMClassifier(c=c_value, p_train=X_train23[y_train == 1].tolist(), n_train=X_train23[y_train == -1].tolist(), sigma=sigma_value)
            SVM23.fit()
            predict_result23 = SVM23.predict(p_label=2, n_label=3, text_data=test_data)

            prediction = []
            for i in range(len(test_data)):
                vote = {1:0, 2:0, 3:0}
                vote[predict_result12[i]] += 1
                vote[predict_result13[i]] += 1
                vote[predict_result23[i]] += 1

                if vote[1] == vote[2] == vote[3]:
                    prediction.append(0)
                else:
                    predict_result = max(vote, key=vote.get)
                    prediction.append(predict_result)


            correct_count = sum(1 for i in range(len(prediction)) if prediction[i] == test_label[i])
            CR1 = correct_count / len(test_data) * 100
            #print(f"CR1: {CR1:.2f}%")

            SVM12 = SVMClassifier(c=c_value, p_train=X2_train12[y_train == 1].tolist(), n_train=X2_train12[y_train == -1].tolist(), sigma=sigma_value)
            SVM12.fit()
            predict_result12 = SVM12.predict(p_label=1, n_label=2, text_data=test_data2)

            SVM13 = SVMClassifier(c=c_value, p_train=X2_train13[y_train == 1].tolist(), n_train=X2_train13[y_train == -1].tolist(), sigma=sigma_value)
            SVM13.fit()
            predict_result13 = SVM13.predict(p_label=1, n_label=3, text_data=test_data2)

            SVM23 = SVMClassifier(c=c_value, p_train=X2_train23[y_train == 1].tolist(), n_train=X2_train23[y_train == -1].tolist(), sigma=sigma_value)
            SVM23.fit()
            predict_result23 = SVM23.predict(p_label=2, n_label=3, text_data=test_data2)

            prediction = []
            for i in range(len(test_data)):
                vote = {1:0, 2:0, 3:0}
                vote[predict_result12[i]] += 1
                vote[predict_result13[i]] += 1
                vote[predict_result23[i]] += 1

                if vote[1] == vote[2] == vote[3]:
                    prediction.append(0)
                else:
                    predict_result = max(vote, key=vote.get)
                    prediction.append(predict_result)

            correct_count = sum(1 for i in range(len(prediction)) if prediction[i] == test_label[i])
            CR2 = correct_count / len(test_data) * 100
            #print(f"CR2: {CR2:.2f}%")
            CR = (CR1 + CR2) / 2
            print(f"CR: {CR:.2f}%")

            row.append(round(CR, 2))
        table_data.append(row)
    df = pd.DataFrame(table_data, columns=c_values, index=sigma_logs)
    print(df)  # 在終端打印表格
    df.to_csv("svm_results_table.csv")  # 將表格保存到 CSV 文件中


if __name__ == "__main__":
    main()
    os.system("pause")