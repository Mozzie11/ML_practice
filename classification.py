import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

if __name__ == "__main__":

# initialize data
    mean_pos = [2,0]
    cov_pos = [[3,0],[0,3]]
    x_pos = np.random.multivariate_normal(mean_pos, cov_pos, 100)
    mean_neg = [-2,0]
    cov_neg = [[1,0],[0,1]]
    x_neg = np.random.multivariate_normal(mean_neg, cov_neg, 100)

# plot data
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    dot_size = 6
    plt.scatter(x_pos[:,0], x_pos[:,1], color='blue', s=dot_size)
    plt.scatter(x_neg[:,0], x_neg[:,1], color='red', s=dot_size)
    plt.show()

# test data
    data = np.vstack((x_pos,x_neg))
    pos_len = x_pos.shape[0]
    neg_len = x_neg.shape[0]
    label = []
    for _ in range(pos_len):
        label.append(1)
    for _ in range(100):
        label.append(neg_len)
    clf = SGDClassifier(loss="log",tol=0.001)
    clf.fit(data,label)
    n_clf = KNeighborsClassifier(n_neighbors=5)
    n_clf.fit(data,label)

    X = data
    y = label
    h = 0.02
    x_min, x_max = -8,8 #X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = -8,8 #X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

# Create color maps
    cmap_light = ListedColormap(['pink', 'blue', 'azure'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# linear classification
    Z = clf.predict(np.c_[xx.flat, yy.flat])
    Z = Z.reshape(xx.shape)
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=6)
    plt.show()

# knn classification
    Z = n_clf.predict(np.c_[xx.flat, yy.flat])
    Z = Z.reshape(xx.shape)
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=6)
    plt.show()