import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
#from metrice import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data
y = digits.target

some_digit = X[666]
some_digit_image = some_digit.reshape(8,8)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)

# =============================================================================
# # 对数据进行分离
# from KNN import train_test_split
# from KNNClassifier import KNNClassifier
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
# 
# my_knn_clf = KNNClassifier(k=3)
# my_knn_clf.fit(X_train,y_train)
# y_predict = my_knn_clf.predict(X_test)
# 
# accuracy_score(y_test,y_predict)
# =============================================================================
# 使用sklearn进行预测
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=666)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_predict))
knn_clf.score(X_test, y_test)