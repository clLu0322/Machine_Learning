from sklearn.datasets import load_breast_cancer
from LDA import LDAClassifier
import numpy as np

def main():
    cancer_data = load_breast_cancer()
    label = cancer_data.target
    data = cancer_data.data

    num_feature = 30
    select_feature = []
    bestCR_history = []
    remain_feature = np.arange(num_feature).tolist() #[0, 1, 2, ......,29]

    for _ in range(num_feature):
        best_CR = -1
        best_feature = None

        for feature in remain_feature:
            current_feature = select_feature + [feature]

            train_data = []
            for data_index in range(569):
                select_data = []
                for feature_index in current_feature: 
                    select_data.append(data[data_index][feature_index])
                train_data.append(select_data)
            
            CR = two_fold_LDA(label, train_data)
            if CR > best_CR:
                best_CR = CR
                best_feature = feature

        
        select_feature.append(best_feature)
        remain_feature.remove(best_feature)
        bestCR_history.append(round(best_CR, 2))
        print(f"Step {_ + 1}: Selected feature {select_feature}, CR: {best_CR:.2f}%")
    
    max_CR = max(bestCR_history)
    max_index = bestCR_history.index(max_CR)

    print(f"Max CR: {max_CR}, Index: {max_index+1}")
    #print(f"Selected features: {select_feature}")
    print(f"CR history: {bestCR_history}")
    # CR = two_fold_LDA(label, train_data)
    # print(f"CR: {CR:.2f}%")




    
def split_data(label=[], feature=[]):
    n_train = []
    p_train = []
    for i in range(len(label)): 
        if label[i] == 0:
            n_train.append(feature[i])
        else:
            p_train.append(feature[i])
    return n_train, p_train



def two_fold_LDA(label, feature):
    a_label = len(label)
    midpoint = a_label // 2  

    forward_label = label[:midpoint]
    forward_feature = feature[:midpoint]
    backward_label = label[midpoint:]
    backward_feature = feature[midpoint:]

    n_train, p_train = split_data(forward_label, forward_feature)
    LDA1 = LDAClassifier(p_label=1, n_label=0, cp=1, cn=1, p_train=p_train, n_train=n_train)
    LDA1.fit()
    prediction = LDA1.predict(backward_feature)
    correct_count = sum(1 for i in range(len(prediction)) if prediction[i] == backward_label[i])
    CR1 = correct_count / len(backward_label) * 100

    n_train, p_train = split_data(backward_label, backward_feature)
    LDA2 = LDAClassifier(p_label=1, n_label=0, cp=1, cn=1, p_train=p_train, n_train=n_train)
    LDA2.fit()
    prediction = LDA2.predict(forward_feature)
    correct_count = sum(1 for i in range(len(prediction)) if prediction[i] == forward_label[i])
    CR2 = correct_count / len(forward_label) * 100

    CR = (CR1 + CR2) / 2
    return CR

main()