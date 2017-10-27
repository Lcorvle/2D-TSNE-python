import csv
import os
import numpy as np
from sklearn.externals import joblib

modelDir = "model/"
clusterPath = "solder_to_classifier"
logDir = "groupdata/"

def getTreeInfo( tree ):
    n_node = tree.node_count
    assert tree.node_count == len( tree.children_left)
    assert ( tree.children_left == -1).sum() == ( tree.children_right == -1 ).sum()
    n_leaf = ( tree.children_right == -1 ).sum()
    n_depth = tree.max_depth
    return n_node, n_leaf, n_depth

def GetAverageInfo( RandomForest ):
    total_node = 0
    total_leaf = 0
    total_depth = 0
    total_tree = 0
    for i in range( RandomForest.n_estimators ):
        node, leaf, depth = getTreeInfo( RandomForest.estimators_[i].tree_)
        total_node += node
        total_leaf += leaf
        total_depth += depth
        total_tree += 1
    total_tree = float( total_tree )
    print( (total_node - total_leaf) / total_tree, float(total_leaf)/total_node, total_leaf / total_tree, total_depth / total_tree )

clusterModels = {}
files= os.listdir(modelDir)
for file in files:
    if not os.path.isdir(file):
        modelPath = modelDir + file
        print( modelPath )
        sld = file.split('_model.txt')[0]
        model = joblib.load(modelPath)
        clusterModels[sld] = model

# print( getTreeInfo( clusterModels['0'].estimators_[0].tree_) )

GetAverageInfo( clusterModels['0'])
GetAverageInfo( clusterModels['1'])
GetAverageInfo( clusterModels['2'])
GetAverageInfo( clusterModels['3'])
GetAverageInfo( clusterModels['4'])