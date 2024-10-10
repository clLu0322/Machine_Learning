from LDA import LDAClassifier, iris_data_split
import os

def main():
    data1_dict, data2_dict = iris_data_split().feature_selection_and_split([2,3])
    test_data = data2_dict[2] + data2_dict[3]
    test_label = [2] * len(data2_dict[2]) + [3] * len(data2_dict[3])

    lda1 = LDAClassifier(p_label = 2, n_label = 3, p_train = data1_dict[2], n_train = data1_dict[3])
    lda1.fit()
    correct_count = 0

    prediction = lda1.predict(test_data)
    for i in range(len(prediction)):
        if prediction[i] == test_label[i]:
            correct_count += 1
            
    CR1 = correct_count/len(test_data)*100
    print(f"分類率: {CR1:.2f}%")

    test_data = data1_dict[2] + data1_dict[3]
    test_label = [2] * len(data1_dict[2]) + [3] * len(data1_dict[3])   
    lda2 = LDAClassifier(p_label = 2, n_label = 3, p_train = data2_dict[2], n_train = data2_dict[3])
    lda2.fit()
    correct_count = 0
    prediction = lda1.predict(test_data)
    for i in range(len(prediction)):
        if prediction[i] == test_label[i]:
            correct_count += 1

    CR2 = correct_count/len(test_data)*100
    print(f"分類率: {CR2:.2f}%")

    print(f"平均分類率:{(CR1+CR2)/2:.2f}%")

if __name__ == "__main__":
    main()
    os.system("pause")