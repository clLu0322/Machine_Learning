from LDA import LDAClassifier
import numpy as np
import copy
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
    CR1 = predict(data1_dict,data2_dict)
    print(f"CR1: {CR1:.2f}%")
    CR2 = predict(data2_dict, data1_dict)
    print(f"CR2: {CR2:.2f}%")

    print(f"Average CR: {(CR1 + CR2 )/ 2 :.2f}%")



def predict(train_dict, test_dict):

    data_1_2 = copy.deepcopy(train_dict)
    data_1_3 = copy.deepcopy(train_dict)
    data_2_3 = copy.deepcopy(train_dict)


    del data_1_2[3]
    del data_1_3[2]
    del data_2_3[1]

    model12= LDAClassifier(p_type = 1, n_type = 2, p_train = train_dict[1], n_train = train_dict[2])
    model12.fit()
    model13= LDAClassifier(p_type = 1, n_type = 3, p_train = train_dict[1], n_train = train_dict[3])
    model13.fit()
    model23= LDAClassifier(p_type = 2, n_type = 3, p_train = train_dict[2], n_train = train_dict[3])
    model23.fit()

    
    correct_count = 0
    test_list = test_dict[1] +test_dict[2] +test_dict[3]
    answer_list = [1] * len(test_dict[1]) + [2] * len(test_dict[2]) + [3] * len(test_dict[3])

    result12 = model12.predict(test_list)
    result13 = model13.predict(test_list)
    result23 = model23.predict(test_list)

    
    prediction = []
    for i in range(len(test_list)):
        vote = {1: 0, 2: 0, 3: 0}
        
        # 計票過程
        vote[result12[i]] += 1
        vote[result13[i]] += 1
        vote[result23[i]] += 1
        
        if vote[1] == vote[2] == vote[3]:
            prediction.append('error')
        else:
            predict_result = max(vote, key=vote.get)
            prediction.append(predict_result)
    
    for i in range(len(prediction)):
        if prediction[i] == answer_list[i]:
            correct_count += 1

    CR = correct_count / len(test_list) *100
    return CR


if __name__ == "__main__":
    main()
            
                
