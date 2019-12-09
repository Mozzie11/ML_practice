
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

if __name__=="__main__":

#logistic回归
    all_data=pd.read_csv('preprocessed_data.csv')
    print(len(all_data))

    training_data=all_data.loc[: int(len(all_data)*0.8),:]
    testing_data=all_data.loc[ int(len(all_data)*0.8):,:]

    model=skl.LogisticRegression(solver='liblinear')
    model.fit(training_data.drop(['Survived'],axis=1),
              training_data['Survived'])
    predict_y=model.predict(training_data.drop(['Survived'],axis=1))
    print('training accuracy:',(
            predict_y == training_data['Survived']).sum()/len(predict_y))

    predict_y=model.predict(testing_data.drop(['Survived'], axis=1))
    print('testing accuracy:', (
            predict_y == testing_data['Survived']).sum()/len(predict_y))

#随机森林

    model=RandomForestClassifier()
    model.fit(training_data.drop(['Survived'],axis=1),
              training_data['Survived'])
    predict_y=model.predict(training_data.drop(['Survived'],axis=1))
    print('training accuracy:',(
            predict_y == training_data['Survived']).sum()/len(predict_y))

    predict_y=model.predict(testing_data.drop(['Survived'], axis=1))
    print('testing accuracy:', (
            predict_y == testing_data['Survived']).sum()/len(predict_y))

