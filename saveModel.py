import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def getTreeInfo( tree ):
    n_node = tree.node_count
    assert tree.node_count == len( tree.children_left)
    assert ( tree.children_left == -1).sum() == ( tree.children_right == -1 ).sum()
    n_leaf = ( tree.children_right == -1 ).sum()
    n_depth = tree.max_depth
    return n_node, n_leaf, n_depth

def getMap( tree ):
    d = {}
    node_count = 0
    leaf_count = -1
    for i in range( len(tree.children_left) ):
        if tree.children_left[i] > -1:
            d[i] = node_count
            node_count += 1
        else:
            d[i] = leaf_count
            leaf_count -= 1
    return d

def getParent( tree,d ):
    p = {}
    for i in range( len(tree.children_left)):
        if tree.children_left[i] > -1 :
            p[d[tree.children_left[i]]]  = d[i]
        if tree.children_right[i] > -1:
            p[d[tree.children_right[i]]] = d[i]
    return p

#load model
clf = joblib.load( './model/0_model.txt')

model_file = open('./model/model_0','w')
#generage model file

    # overall model information
model_file.writelines('tree\n')
model_file.writelines( 'num_class='+str(len(clf.classes_)) + '\n' )
model_file.writelines( 'num_tree_per_iteration='+str(len(clf.classes_))+ '\n'  )
#TODO: is it always 0
model_file.writelines('label_index=0'+ '\n' )
model_file.writelines('max_feature_idx=' + str(clf.n_features_)+ '\n' )
#TODO: cheat this binaryclass as multiclass
model_file.writelines('objective=multiclass num_class:'+ str(len(clf.classes_)) + '\n' )
s = 'feature_names='
for i in range(clf.n_features_):
    s += 'feature_'+str(i) + ' '
model_file.writelines(s[:-1] + '\n')
    #TODO: what is feature infos
model_file.writelines('feature_infos'+ '\n' )

    #single tree information
tree_num = 0
for dctree in clf.estimators_:
    tree = dctree.tree_
    n_node, n_leaf, n_depth = getTreeInfo(tree)
    m = getMap(tree)
    p = getParent(tree, m )
    value = tree.value.reshape( tree.value.shape[0],-1)
    value = value / value.sum(axis=1).reshape(-1,1).repeat(axis=1,repeats=len(clf.classes_))
    for i in range(len(clf.classes_)):
        model_file.writelines('\n')

        model_file.writelines( 'Tree=' + str(tree_num) + '\n' )
        model_file.writelines('num_leaves='+ str(n_leaf) + '\n')
        #TODO : is it always 0
        model_file.writelines('num_cat=0'+ '\n')
        # split_features
        split_feature = 'split_feature='
        for t in tree.feature:
            if t > -1:
                s += str(t) + ' '
        model_file.writelines(split_feature[:-1] + '\n')

        #split_gain
        split_gain = 'split_gain='
        for t in tree.impurity:
            if t > 1e-6:
                split_gain += str(t) + ' '
        model_file.writelines(split_gain[:-1] + '\n')

        #threshold
        threshold = 'threshold='
        for t in tree.threshold:
            if  t > -1:
                threshold += str(t) + ' '
        model_file.writelines(threshold[:-1] + '\n')

        #decision type
        decision_type = 'decision_type='
        for t in tree.feature:
            if t > -1:
                decision_type += str(t) + ' '
        model_file.writelines( decision_type[:-1] + '\n' )

        #child and parents
        left_child = 'left_child='
        right_child = 'right_child='
        leaf_parent = 'leaf_parent='
        for t in range( len(tree.children_left) ):
            if tree.children_left[t] > -1:
                left_child += str( m[tree.children_left[t]] ) + ' '
                right_child += str( m[tree.children_right[t]]) + ' '
        for t in range(1,n_leaf+1):
            leaf_parent += str( p[-t] ) + ' '
        model_file.writelines( left_child[:-1] + '\n')
        model_file.writelines( right_child[:-1] + '\n')
        model_file.writelines( leaf_parent[:-1] + '\n' )

        # value and count
        leaf_value = 'leaf_value='
        leaf_count = 'leaf_count='
        internal_value = 'internal_value='
        internal_count = 'internal_count='
        for t in range( n_node ):
            # internal_node
            if m[t] > -1:
                internal_value += str( value[t,i]) + ' '
                internal_count += str( tree.n_node_samples[t] ) + ' '
            #leaf_node
            else:
                leaf_value += str( value[t,i]) + ' '
                leaf_count += str( tree.n_node_samples[t] ) + ' '

        model_file.writelines( leaf_value[:-1] + '\n' )
        model_file.writelines( leaf_count[:-1] + '\n' )
        model_file.writelines( internal_value[:-1] + '\n' )
        model_file.writelines( internal_count[:-1] + '\n' )

        model_file.writelines('\n')
        tree_num += 1
    # break

    #feature importances:
model_file.writelines('\n\nfeature_importances:\n')
FI = np.array(clf.feature_importances_)
order = FI.argsort()
for i in order[::-1]:
    model_file.writelines('feature_' + str(i) + '=' + str(FI[i]) + '\n' )

    #pandas_categorical:null
model_file.writelines('\npandas_categorical:null\n')
model_file.close()

None