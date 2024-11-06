import numpy as np

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




    