#only regression
import pandas as pd
import numpy as np

df = pd.read_csv("/home/multipodo/bkpchristian/cheef/iris.csv")
columna = [lista for lista in df.columns]
columna.remove(columna[len(columna) - 1])
columna.append('Decision')
df.columns = columna

rows=df.shape[0]
diccionario = {}
antidiccionario = {}
for column in range(len(df.columns)):
    lista=[]
    lista_={}
    antilista_={}
    cont=0
    if df[df.columns[column]].dtypes == 'object':
        for row in range(rows):
            if (not df.iloc[row,column] in lista):
                lista_[df.iloc[row,column]]=cont
                antilista_[str(cont)] = df.iloc[row,column]
                lista.append(df.iloc[row,column])
                cont+=1
        diccionario[df.columns[column]]=lista_
        antidiccionario[df.columns[column]]=antilista_
def to_number(df1):     
    for column in range(len(df1.columns)):
        if df1[df1.columns[column]].dtypes == 'object':
            for row in range(rows):
                df1.iloc[row,column] = diccionario[df1.columns[column]][df1.iloc[row,column]]
    df1 = df1.astype('float32')
    return df1

def isNumeric(x):
    try:
        numero = float(x)
        n=True
    except:
        n=False
    return n

def features(feature1):
    feature_=[]
    for column in range(len(feature1)):
        if(not isNumeric(feature1[column])):

            feature_.append(diccionario[df.columns[column]][feature1[column]])
        else:
            feature_.append(feature1[column])

    return feature_

df=to_number(df)
df2 = df.copy()

#Regression
from chefboost import Chefboost as chef
config = {'algorithm': 'Regression'}
model = chef.fit(df, config)
#feature_=['Overcast','Cool','Normal','Strong']
feature_=[1,2,3,4]
feature = features(feature_)

prediction = chef.predict(model, feature)
print(feature_,antidiccionario[df2.columns[len(df2.columns)-1]][str(round(prediction))])
count=0
for index, instance in df2.iterrows():

    feature = features(instance)
    #print(index, feature)
    prediction = antidiccionario[df2.columns[len(df2.columns)-1]][str(round(chef.predict(model, feature)))]
    actual = antidiccionario[df2.columns[len(df2.columns)-1]][str(round(float(instance['Decision'])))]
    print(index+1,'\tActual:', actual,'\t- \tPredict',prediction,'\tmatch: ', prediction==actual)
    if(prediction==actual):
        count+=1
print('match',str(round(count/df.shape[0]*100,2)))