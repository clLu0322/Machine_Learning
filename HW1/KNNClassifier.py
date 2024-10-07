import os
# K-NN 分類器
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []  # 訓練資料
        self.y_train = []  # 訓練標籤

    # 訓練模型
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 預測新數據
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i in range(len(self.X_train)):
                distance = euclidean_distance(test_point, self.X_train[i])
                distances.append((distance, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            # 手動計算最常見的標籤
            neighbor_labels = [neighbor[1] for neighbor in neighbors]
            most_common = self.most_common_label(neighbor_labels)
            predictions.append(most_common)
        return predictions

    # 計算準確度
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(p == y for p, y in zip(predictions, y_test))
        return correct / len(y_test)

    # 找到列表中出現最多次的標籤
    def most_common_label(self, labels):
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return max(label_counts, key=label_counts.get)

# 讀取資料並輸出字典
def load_iris_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            data.append(parts[:-1])  # 特徵
            labels.append(int(parts[-1]))  # 標籤
    return {"data": data, "labels": labels}

# 計算歐幾里得距離
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

# 按類別分前後兩半
def two_fold_cross_validation(X, y, k=1):
    class_data = {}
    for i in range(len(y)):
        label = y[i]
        if label not in class_data:
            class_data[label] = {'data': [], 'labels': []}
        class_data[label]['data'].append(X[i])
        class_data[label]['labels'].append(y[i])

    X_train, y_train = [], []
    X_test, y_test = [], []

    # 將每個類別的資料分成前後兩半
    for label, data_dict in class_data.items():
        data, labels = data_dict['data'], data_dict['labels']
        n = len(data) // 2
        X_train.extend(data[:n])
        y_train.extend(labels[:n])
        X_test.extend(data[n:])
        y_test.extend(labels[n:])

    # 第一次分類，訓練集是前半部分，測試集是後半部分
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    score1 = model.score(X_test, y_test)

    # 第二次分類，互換訓練集和測試集
    model.fit(X_test, y_test)
    score2 = model.score(X_train, y_train)
    return (score1 + score2) / 2

# 手動生成所有特徵組合
def generate_combinations(num_features):
    combinations = []
    # 生成 1 至 num_features 個特徵的所有組合
    for i in range(1, 2 ** num_features):
        combo = []
        for j in range(num_features):
            if i & (1 << j):
                combo.append(j)
        combinations.append(combo)
    return combinations

# 計算所有可能特徵組合的分類率
def feature_combinations_classification(X, y, k=1):
    num_features = len(X[0]) 
    results = []

    # 手動生成所有特徵組合
    combinations = generate_combinations(num_features)
    for combo in combinations:
        X_subset = [[x[i] for i in combo] for x in X]
        score = two_fold_cross_validation(X_subset, y, k=k) * 100
        results.append((combo, round(score, 2)))

    # 設置標題並定義列寬
    print(f"{'Feature combination'.ljust(25)} | {'Classification Rate (%)'.rjust(25)}")
    print("-" * 55)

    # 列印結果並對齊
    for combo, score in results:
        combo_str = str(combo).ljust(25)
        score_str = f"{score}%".rjust(25)
        print(f"{combo_str} | {score_str}")

# 主程序
def main():
    dataset = load_iris_data('iris.txt')
    X = dataset['data']
    y = dataset['labels']

    k_value = int(input("請輸入 K 值: "))
    print(f"\nK = {k_value} 的特徵組合分類結果:")
    feature_combinations_classification(X, y, k=k_value)

if __name__ == "__main__":
    main()
    os.system("pause")
