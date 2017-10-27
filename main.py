import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

from src.tsne import loadData, CV_stratified_slice, dataset_concatenate

def train_randomforest( X, y, test_X, test_y, n_estimators=10):
    clf = RandomForestClassifier( n_estimators=n_estimators, class_weight='balanced')
    clf.fit( X, y )
    pred_y = clf.predict(test_X)
    acc = accuracy_score(test_y,pred_y )
    recall_pos = recall_score(test_y, pred_y )
    recall_neg = recall_score( 1 - test_y, 1 - pred_y )
    precision_pos = precision_score( test_y, pred_y )
    precision_neg = precision_score( 1 - test_y, 1 - pred_y )
    return acc, recall_pos, recall_neg, precision_pos, precision_neg

def main():
    data = loadData('../../Ximei/pureTrainingData/model_81_full.csv')
    X = data["X"]
    y = data["y"]
    res = CV_stratified_slice(X, y, 5)
    for i in range(len(res)):
        rest_data = [ res[j] for j in range(len(res)) if j != i ]
        rest_X, rest_y = dataset_concatenate(rest_data)
        acc, recall_pos, recall_neg, precision_pos, precision_neg = train_randomforest( rest_X, rest_y, res[i]["X"], res[i]["y"] )
        print( "cv %s: acc %.5f,  recall_pos %.5f, recall_neg %.5f, precision_pos %.5f, precision_neg %.5f" %( i, acc, recall_pos, recall_neg, precision_pos, precision_neg ) )

if __name__ == '__main__':
    main()