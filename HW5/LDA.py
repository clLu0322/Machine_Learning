import numpy as np

class LDAClassifier:
    def __init__(self, p_label, n_label, cp=1, cn=1, p_train=[], n_train=[]):
        self.cn = cn
        self.cp = cp
        self.p_label = p_label
        self.n_label = n_label
        self.p_train_data = np.array(p_train)
        self.n_train_data = np.array(n_train)

    def fit(self):
        self.p_amount = len(self.p_train_data)
        self.n_amount = len(self.n_train_data)

        self.p_proportion = self.p_amount / (self.p_amount + self.n_amount)
        self.n_proportion = self.n_amount / (self.p_amount + self.n_amount)

        self.p_mean = np.mean(self.p_train_data, axis=0)
        self.n_mean = np.mean(self.n_train_data, axis=0)

        p_covariance = np.cov(self.p_train_data.T, bias=False)
        n_covariance = np.cov(self.n_train_data.T, bias=False)

        if p_covariance.ndim == 0:  # 如果是标量，转换为二维矩阵
            p_covariance = np.array([[p_covariance]])
        if n_covariance.ndim == 0:  # 如果是标量，转换为二维矩阵
            n_covariance = np.array([[n_covariance]])

        self.co_variance_matrix = self.p_proportion * p_covariance + self.n_proportion * n_covariance
        cov_inv = np.linalg.inv(self.co_variance_matrix)
        self.w = np.dot(cov_inv, (self.p_mean - self.n_mean).T)
        self.b = -0.5 * np.dot(self.w.T, (self.p_mean + self.n_mean)) - np.log(self.cp * self.p_proportion / self.cn / self.n_proportion)


    def predict(self, test_data=[]):
        test_data = np.array(test_data)

        result = []

        for i in range(len(test_data)):
            prediction = np.dot(self.w.T, test_data[i]) + self.b
            if prediction >= 0:
                result.append(self.p_label)
            else:
                result.append(self.n_label)

        return result
    

# def main():
#     p_train = [
#         [1, 2],
#         [2, 5],
#         [3, 2]
#     ]

#     n_train = [
#         [-4, -7],
#         [-3, -2],
#         [-2, 0]
#     ]

#     LDA = LDAClassifier(p_label=1, n_label=0, cp=1, cn=1, p_train=p_train, n_train=n_train)
#     LDA.fit()
    
# main()