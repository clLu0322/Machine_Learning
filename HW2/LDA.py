import numpy as np

class LDAClassifier:
    def __init__(self, p_label, n_label, cp=1, cn=1, p_train=[], n_train = []):
        self.cn = cn
        self.cp = cp
        self.p_label = p_label
        self.n_label = n_label
        self.p_train_data = p_train
        self.n_train_data = n_train

    def fit(self):
        self.p_amount = len(self.p_train_data)
        self.n_amount = len(self.n_train_data)

        self.p_proportion = self.p_amount / (self.p_amount + self.n_amount)
        self.n_proportion = self.n_amount / (self.p_amount + self.n_amount)

        self.p_mean = np.mean(np.array(self.p_train_data), axis=0)
        self.n_mean = np.mean(np.array(self.n_train_data), axis=0)

        self.dim = self.p_train_data[0].shape
        self.co_variance_matrix = np.zeros((self.dim[0], self.dim[0]))
        p_covariance = np.zeros((self.dim[0], self.dim[0]))
        n_covariance = np.zeros((self.dim[0], self.dim[0]))
        for i in range(self.p_amount):
            diff = self.p_train_data[i] - self.p_mean
            p_covariance += np.dot(diff, diff.T)
        p_covariance /= (self.p_amount - 1)

        for i in range(self.n_amount):
            diff = self.n_train_data[i] - self.n_mean
            n_covariance += np.dot(diff, diff.T)
        n_covariance /= (self.n_amount - 1)

        self.co_variance_matrix = self.p_proportion * p_covariance + self.n_proportion * n_covariance

        self.w = np.round(np.dot((self.p_mean - self.n_mean).T, np.linalg.inv(self.co_variance_matrix)), decimals=2)
        self.b = np.round((-1/2) * np.dot(self.w, (self.p_mean + self.n_mean)) - np.log(self.cp * self.p_proportion / self.cn / self.n_proportion), decimals=2)

        print(f"權重向量 w: {self.w}")
        print(f"偏置項 b: {self.b}")

    def predict(self, test_data = []):
        result = []
        for i in range(len(test_data)):
            if (test_data[i].shape != self.dim):
                result.append('error')
            else:
                prediction = np.dot(self.w, test_data[i]) + self.b
                if prediction >= 0:
                    result.append(self.p_label)
                else:
                    result.append(self.n_label)
        return result
    
    def get_scores(self, test_data = []):
        scores = []
        for i in range(len(test_data)):
            if (test_data[i].shape != self.dim):
                print('error')
            else:
                scores.append((np.dot(self.w, test_data[i]) + self.b).item())  
        return scores


class iris_data_split:
    def __init__(self):
        self.data = np.loadtxt('iris.txt')

    def feature_selection_and_split(self, selected_features):
        
        # 提取所有特徵
        features = [self.data[:, i] for i in range(self.data.shape[1] - 1)]
        categories = self.data[:, -1].astype(int) 
        data1_dict = {} #前半
        data2_dict = {} #後半

        for i in range(len(categories)):
            category = categories[i]
            # 根據選擇的特徵組成矩陣，每個特徵都是一行
            selected_matrix = np.array([[features[j][i]] for j in selected_features])
            if i % 50 < 25:
                if category not in data1_dict:
                    data1_dict[category] = []
                    data2_dict[category] = []
                data1_dict[category].append(selected_matrix)
            else:
                data2_dict[category].append(selected_matrix)

        return data1_dict, data2_dict
    """
    data format
    {1:
    [[1.2],
     [2.2],
     [2.7]]
     2:
     [[1.2],
     [2.2],
     [2.7]]
     }
    """