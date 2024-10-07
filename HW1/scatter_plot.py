import matplotlib.pyplot as plt
import numpy as np

# 從 iris.txt 文件中讀取數據
data = np.loadtxt('iris.txt')

# 提取四個屬性和類別
sepal_length = data[:, 0]
sepal_width = data[:, 1]
petal_length = data[:, 2]
petal_width = data[:, 3]
species = data[:, 4]  # 類別 (1, 2, 3)

# 定義類別和對應的顏色
colors = {1: 'blue', 2: 'green', 3: 'red'}
labels = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}

# 散佈圖組合列表
combinations = [
    (sepal_length, sepal_width, 'Sepal Length', 'Sepal Width'),
    (sepal_length, petal_length, 'Sepal Length', 'Petal Length'),
    (sepal_length, petal_width, 'Sepal Length', 'Petal Width'),
    (sepal_width, petal_length, 'Sepal Width', 'Petal Length'),
    (sepal_width, petal_width, 'Sepal Width', 'Petal Width'),
    (petal_length, petal_width, 'Petal Length', 'Petal Width')
]

# 繪製六個散佈圖，按類別顯示不同顏色
plt.figure(figsize=(10, 12))
for i, (x, y, xlabel, ylabel) in enumerate(combinations, 1):
    plt.subplot(3, 2, i)
    for class_label in np.unique(species):
        plt.scatter(x[species == class_label], y[species == class_label], 
                    c=colors[class_label], label=labels[class_label], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{xlabel} vs {ylabel}')
    plt.legend()

plt.tight_layout()
plt.show()
