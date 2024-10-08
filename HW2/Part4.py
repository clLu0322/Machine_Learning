from LDA import LDAClassifier, iris_data_split

def main():
    data1_dict, data2_dict = iris_data_split().feature_selection_and_split([2,3])
    CR1 = predict(data1_dict,data2_dict)
    print(f"CR1: {CR1:.2f}%")
    CR2 = predict(data2_dict, data1_dict)
    print(f"CR2: {CR2:.2f}%")
    print(f"Average CR: {(CR1 + CR2 )/ 2 :.2f}%")

def predict(train_dict, test_dict):

    #訓練所有模型
    model12= LDAClassifier(p_label = 1, n_label = 2, p_train = train_dict[1], n_train = train_dict[2])
    model12.fit()
    model13= LDAClassifier(p_label = 1, n_label = 3, p_train = train_dict[1], n_train = train_dict[3])
    model13.fit()
    model23= LDAClassifier(p_label = 2, n_label = 3, p_train = train_dict[2], n_train = train_dict[3])
    model23.fit()

    correct_count = 0
    test_list = test_dict[1] +test_dict[2] +test_dict[3]
    test_label = [1] * len(test_dict[1]) + [2] * len(test_dict[2]) + [3] * len(test_dict[3])

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
        if prediction[i] == test_label[i]:
            correct_count += 1

    CR = correct_count / len(test_list) *100
    return CR

if __name__ == "__main__":
    main()
            
                
