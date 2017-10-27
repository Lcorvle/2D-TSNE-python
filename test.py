from sklearn.tree import DecisionTreeClassifier as DTC

X = [[0],[1],[2]] # 3 simple training examples
Y = [ 1,  2,  1 ] # class labels

dtc = DTC(max_depth=1)

dtc.fit(X,Y,sample_weight=[1.5,2,2.5])
print(dtc.tree_.threshold)
print(dtc.tree_.impurity)