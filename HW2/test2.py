from test import LDAClassifier
import numpy as np

def data():
    data = np.loadtxt('iris.txt')

    feature3 = data[:, 2]
    feature4 = data[:, 3]
    categories = data[:, -1].astype(int)
    data1_dict = {}
    data2_dict = {}
    for i in range(len(categories)):
        category = categories[i]
        matrix = np.array([[feature3[i]], [feature4[i]]])
        
        if i % 50 <25:
            if category not in data1_dict:
                data1_dict[category] = []
                data2_dict[category] = []
            data1_dict[category].append(matrix)
        else:
            data2_dict[category].append(matrix)

    return data1_dict, data2_dict

def main():
    data1_dict, data2_dict = data()
    test_data = data2_dict[2] + data2_dict[3]
    answer_list = []
    answer_list = [2] * len(data2_dict[2]) + [3] * len(data2_dict[3])

    lda1 = LDAClassifier(p_type = 2, n_type = 3, p_train = data1_dict[2], n_train = data1_dict[3])
    lda1.fit()
    correct_count = 0

    prediction = lda1.predict(test_data)
    for i in range(len(prediction)):
        if prediction[i] == answer_list[i]:
            correct_count += 1

    
    CR1 = correct_count/len(test_data)*100
    print(f"分類率: {CR1:.2f}%")

    test_data = data1_dict[2] + data1_dict[3]
    answer_list = [2] * len(data1_dict[2]) + [3] * len(data1_dict[3])   
    lda2 = LDAClassifier(p_type = 2, n_type = 3, p_train = data2_dict[2], n_train = data2_dict[3])
    lda2.fit()
    correct_count = 0
    prediction = lda1.predict(test_data)
    for i in range(len(prediction)):
        if prediction[i] == answer_list[i]:
            correct_count += 1

    CR2 = correct_count/len(test_data)*100
    print(f"分類率: {CR2:.2f}%")

    print(f"平均分類率:{(CR1+CR2)/2:.2f}%")

if __name__ == "__main__":
    main()