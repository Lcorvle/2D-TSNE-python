import numpy as np
import pandas
dataPath = '../trainingData/model_1.csv'

d = np.array( pandas.read_csv( '../trainingData/model_1.csv' ) )
d = d[ d[:, -1] < 100, :]

