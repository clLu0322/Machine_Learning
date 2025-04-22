from sklearn.datasets import load_breast_cancer
from LDA import LDAClassifier
import numpy as np

def main():
    cancer_data = load_breast_cancer()
    label = cancer_data.target
    data = cancer_data.data

    fisher_score_list = []
    for feature in range(30):
        train_data = []
        for data_index in range(569):
            select_data = data[data_index][feature]
            train_data.append(select_data)

        n_train, p_train = split_data(label, train_data)
        fisher_score_list.append(cal_fisher_score(n_train, p_train))
    
    print(fisher_score_list)

    sorted_data_with_index = sorted(enumerate(fisher_score_list), key=lambda x: x[1], reverse=True)
    feature_rank = [x[0] for x in sorted_data_with_index]  # 原始索引
    fisher_score_sort = [x[1] for x in sorted_data_with_index]   # 排序後的值

    N = 0
    best_CR = 0
    best_CR_index = -1
    CR_history = []
    current_feature = []
    for feature in feature_rank:
        N += 1
        current_feature = current_feature + [feature]
        train_data = []
        for data_index in range(569):
            select_data = []
            for feature_index in current_feature: 
                select_data.append(data[data_index][feature_index])
            train_data.append(select_data)   

        CR = two_fold_LDA(label, train_data)

        if(CR > best_CR):
            best_CR = CR
            best_CR_index = N

        CR_history.append(round(CR, 2))
        print(f"N = {N}, feature: {current_feature}, CR: {CR:.2f}%")
    print(CR_history)
    print(f"N:{best_CR_index}, best_CR:{best_CR:.2f}")




def cal_fisher_score(n_train=[], p_train=[]):
    if len(n_train) == 0 or len(p_train) == 0:
        raise ValueError("Training data for one of the classes is empty.")

    n_amount = len(n_train)
    p_amount = len(p_train)

    n_p = n_amount / (n_amount + p_amount)
    p_p = p_amount / (n_amount + p_amount)

    n_mean = np.mean(n_train)
    p_mean = np.mean(p_train)
    total_mean = np.mean(np.concatenate((n_train, p_train)))

    # Initialize within-class scatter
    n_sw = 0
    p_sw = 0
    
    # Calculate within-class scatter
    n_sw = np.sum((n_train - n_mean) ** 2) 
    p_sw = np.sum((p_train - p_mean) ** 2)
    
    sw = (n_sw + p_sw) / (n_amount + p_amount)
    
    # Calculate between-class scatter
    sb = n_p * (n_mean - total_mean) ** 2 + p_p * (p_mean - total_mean) ** 2
    
    # Fisher Score
    fisher_score = sb / sw
    return round(float(fisher_score), 4)

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