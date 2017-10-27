import numpy as np
import json

from sklearn.cluster import KMeans

def kmeans_clustering(score, k):
	kmeans = KMeans(n_clusters=k, random_state=0).fit(score)
	labels = kmeans.labels_.tolist()
	centroids = kmeans.cluster_centers_.tolist()
	cluster_size = np.bincount(labels).tolist()
	return centroids, labels, cluster_size

js = []

with open('model.json') as json_file:
    js = json.load(json_file)

data = []
for i in range( len(js) ):
    data.append( js[i]["leaf_count"] )

data = np.array(data)

#TODO:随机产生中心点
centroid = (np.random.rand(5) * 1000 ).astype(int).tolist()

#计算每个点到各个中心点的距离
dis = np.zeros( ( data.shape[0], len(centroid) ) )
for i in range( len(centroid) ):
    center = data[centroid[i]]
    dis[:,i] = ( (data - center.reshape(1,-1).repeat(axis=0,repeats=data.shape[0]) )**2 ).sum(axis=1)

cluster_size = []
cluster_inst_set = []

for i in range( len(centroid ) ):
    cluster_size.append(0)
    cluster_inst_set.append([])

#得到每个点属于哪个中心点的label
labels = dis.argmax(axis=1)

for i in range( data.shape[0]):
    cluster_size[labels[i]] += 1
    cluster_inst_set[labels[i]].append(i)

#TODO: 得到average tree size
cluster_average_tree_size = [3,3,3,3,3]

res = {
    'medoids': centroid,
    'cluster_average_tree_size': cluster_average_tree_size,
    'cluster_inst_set': cluster_inst_set
}

with open('model_clustering_xgboost.json', 'w') as json_file:
    json_file.write(json.dumps([res]))

None
