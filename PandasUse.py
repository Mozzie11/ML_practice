import pandas as pd
import numpy as np

if __name__=="__main__":

#DataFrame是表格数据
#是一种二维标记数据结构
#具有可能不同类型的列。
#Series是一维标记数组
#能够保存任何数据类型（整数，字符串，浮点数，Python对象等
#轴标签统称为索引。 数据帧可以被视为由几个系列组成，每个系列作为一列。

    s=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
    s2=pd.Series([6,7,8,9,10],index=['a','b','c','d','e'])
#    print(s)
#    print(s2[1])
#    print(s2['a'])

#这里介绍了两种创建新数据帧的方法：
#来自numpy数组或来自Series的list / dict
#而不是在Numpy中使用位置进行索引，您可以自定义不同类型的索引。
#对于行，通常我们使用可以唯一标识样本的字符串/整数变量，
#例如student id。 对于列，我们使用要素名称作为索引。

    dataframe1=pd.DataFrame(np.ones((6,4)),index=['one','two','three','four','five','six'],
                 columns=['A','B','C','D'])
#    print(dataframe1)

#pandas对象中的索引可以使用已知指示符索引/选择数据，并启用自动和显式数据对齐。
    dataframe2=pd.DataFrame([s,s2],index=None,columns=['a','b','c','d','e'])
#    print(dataframe2)

#Pandas提供了使用不同类型创建DataFrame的工具，它们可用于创建DataFrame结构。（涉及广播）
#concat函数是在pandas底下的方法，可以将数据根据不同的轴作简单的融合
#axis： 需要合并链接的轴，0是行，1是列 join：连接的方式 inner，或者outer
    dataframe3=pd.concat([s,s2],axis=1)
#    print(dataframe3)

    df=pd.DataFrame({'A':1.,
                     'B':pd.Timestamp('20130102'),
                     'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D':np.array([3]*4),
                     'E':pd.Categorical(['test','train','test','train']),
                     'F':'f00'})
#    print(df)

    trainDataframe=pd.read_csv('train.csv')
    trainDataframe=trainDataframe.set_index('PassengerId')
#    print(trainDataframe)

    Age=trainDataframe['Age']
#    print(Age)

#    print(trainDataframe.columns)
#    print(trainDataframe.index)
#    print(trainDataframe.dtypes)#相当于sql中每一个键它的类型

    head=trainDataframe.head()#默认前5个
#    print(head)
    tail=trainDataframe.tail(10)
#    print(tail)

#describe()显示数字要素的快速统计摘要 例如均值 最大最小值 总和等等
    AgeDescribe=trainDataframe['Age'].describe()
#    print(AgeDescribe)

#Value_counts对分类功能执行计数
    EmbarkedCount=trainDataframe['Embarked'].value_counts()
#    print(EmbarkedCount)

#从pandas中选择/索引数据与numpy中的数据类似。 这里提供了两种方法。
#dataFrame.loc [“row_index_name”，“column_index_name”]
#dataFrame.iloc [“row_id”，“column_id”]
#loc是根据dataframe的具体标签选取列，而iloc是根据标签所在的位置，从0开始计数。

#    print(trainDataframe.loc[1,'Name'])
#    print(trainDataframe.loc[1])
#    print(trainDataframe.loc[1:5])

#    print(trainDataframe.iloc[0,0])
#    print(trainDataframe.iloc[0])
#    print(trainDataframe.iloc[0:3,0:3])

#    print(trainDataframe["Sex"][1])#这里0无意义 1作为第一个

#选取特定条件用布尔索引#***（请注意，它现在从行索引开始！）

#    print(trainDataframe[trainDataframe['Age']>20]['Age'])#注意括号
#    print(trainDataframe.loc[trainDataframe["Age"] > 20,"Age"])

#增加删除#Inplace = True：没有返回DataFrame，更改发生在trainDataframe中。
#Inplace = False：返回更改的DataFrame，在trainDataframe中不会发生更改。
#默认值：Inplace = False。

    trainDataframe.drop(['Ticket','Name'],axis=1,inplace=True)
#    print(trainDataframe)

    trainDataframe['FamilySize']=trainDataframe['Parch']+trainDataframe['SibSp']
#    print(trainDataframe)

#排序
    sort=trainDataframe.sort_values(by='Fare',ascending=False)#上升错误即是下降
#    print(sort)

#pandas主要使用值np.nan来表示缺失的数据。 默认情况下，它不包含在计算中。
#有两种方法可以处理丢失的数据：删除包含nan值的行或使用默认值填充它。

#    print(trainDataframe.isnull().sum(axis=0))

#    print(trainDataframe.dropna(how='any'))#默认axis=0 这里any是指任何一行有空值就把此值删去

#使用指定的方法填充NA/NaN值
#trainDataframe.fillna(value={"Cabin": "Unknown",
# "Age": trainDataframe["Age"].mean()}).dropna(how="any").isnull().sum(axis = 0)
#年龄缺失的地方填写均值 cabin缺失填写unknow

    trainDataframe = trainDataframe.fillna(value={"Cabin": "Unknown",
                                              "Age": trainDataframe["Age"].mean(),
                                              "Embarked":trainDataframe["Embarked"].mode()}).dropna(how="any")
#    print(trainDataframe)

#apply函数对列中的每个元素应用一个函数。例子中的函数是取第一个值
#    print(trainDataframe['Embarked'].mode())
    trainDataframe["Cabin"] = trainDataframe["Cabin"].apply(lambda x : x[0])
#    print(trainDataframe)

#一键编码 对于不是序数的分类特征，我们倾向于将其转换为一键编码特征的列表

#    print(pd.get_dummies(trainDataframe[['Sex','Cabin','Embarked']]))

#merge合并

    trainDataframe=pd.concat([trainDataframe,pd.get_dummies(trainDataframe[['Sex','Cabin','Embarked']])],axis=1)
    trainDataframe=trainDataframe.drop(['Sex','Cabin','Embarked'],axis=1)
#    print(trainDataframe)

#归一化简化了模型在学习权重方面的工作
    features=trainDataframe.loc[:,trainDataframe.columns !='Survived']
    trainDataframe.loc[:,trainDataframe.columns !='Survived']\
        =(features-features.mean())/features.std()#计算矩阵标准差
#    print(trainDataframe)


    trainDataframe.to_csv('preprocessed_data.csv')
#889rows*21columns
