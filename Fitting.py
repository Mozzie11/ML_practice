import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#拟合不足与拟合过度
#我们可以看到线性函数（阶数为1的多项式）不足以拟合训练样本。这称为欠拟合。
#4次多项式几乎完美地逼近了真函数。但是，对于较高的度数，模型将过度拟合训练数据，即模型会学习训练数据的噪声。
#我们定量评估使用交叉验证来过度拟合 / 不足拟合。
#我们在验证集上计算均方误差（MSE），数值越高，模型从训练数据中正确推广的可能性就越小。

def true_fun(X):
    return np.cos(1.5*np.pi*X)

if __name__=="__main__":
    np.random.seed(0)

    n_samples=30
    degree=15

    X=np.sort(np.random.rand(n_samples))
    y=true_fun(X)+np.random.randn(n_samples)*0.1

    X2=np.sort(np.random.rand(n_samples))
    y2=true_fun(X2)+np.random.randn(n_samples)*0.1

    plt.figure(dpi=300)
    ax=plt.subplot(1,1,1)
    plt.setp(ax,xticks=(),yticks=())

    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression()
    pipeline=Pipeline([('polynomial_features',polynomial_features),
                       ('linear_regression',linear_regression)])
    pipeline.fit(X[:,np.newaxis],y)

#用交叉验证检测模型

    scores=cross_val_score(pipeline,X[:,np.newaxis],y,
                           scoring='neg_mean_squared_error',cv=10)

    X_test=np.linspace(0,1,100)
    plt.plot(X_test,true_fun(X_test),label='True function',color='red')
    plt.plot(X_test,pipeline.predict(X_test[:,np.newaxis]),label='Model')
    plt.scatter(X,y,color='b',s=10,label='Unseen samples')
    plt.scatter(X2,y2,color='g',s=10,label='Unseen samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((0,1))
    plt.ylim((-2,2))
    plt.legend(loc='best')
    plt.show()

#
    n_samples = 30
    degree=4

    plt.figure(dpi=300)
    ax = plt.subplot(1, 1, 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

# Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                         scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, true_fun(X_test), label="True function", color="red")
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.scatter(X, y, color='b', s=10, label="Unseen samples")
    plt.scatter(X2, y2, color='g', s=10, label="Unseen samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.show()

