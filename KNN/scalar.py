#==========================【数据归一化】========================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
#from metrice import accuracy_score
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

X = iris.data
y = iris.target 

# =============================================================================
# # 最值归一化
# # 一维
# x = np.random.randint(0, 100, size=100)
# 
# (x - np.min(x))/(np.max(x)-np.min(x))
# 
# # 二维矩阵
# X = np.random.randint(0, 100, (50, 2))
# X = np.array(X, dtype=float)
# X[:, 0] = (X[:, 0] - np.min(X[:, 0]))/(np.max(X[:, 0]) - np.min(X[:, 0]))
# X[:, 1] = (X[:, 1] - np.min(X[:, 1]))/(np.max(X[:, 1]) - np.min(X[:, 1]))
# 
# 
# 
# # 均值方差归一化
# 
# X2 = np.random.randint(0, 100, (50, 2))
# X2 = np.array(X2, dtype=float)
# 
# 
# X2[:, 0] = (X2[:, 0] - np.mean(X2[:, 0]))/np.std(X2[:, 0])
# 
# X2[:, 1] = (X2[:, 1] - np.mean(X2[:, 1]))/np.std(X2[:, 1])
# 
# plt.scatter(X2[:,0],X2[:,1])
# 
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=666)

###均值方差归一化

from sklearn.preprocessing import StandardScaler
standardScalar = StandardScaler()
standardScalar.fit(X_train)

standardScalar.mean_
standardScalar.scale_

X_train = standardScalar.transform(X_train)
X_test = standardScalar.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)

knn_clf.fit(X_train, y_train)

knn_clf.score(X_test, y_test)
















