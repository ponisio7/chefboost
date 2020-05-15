#no regression
import pandas as pd
import numpy as np

df = pd.read_csv("/home/multipodo/bkpchristian/cheef/iris.csv")
columna = [lista for lista in df.columns]
columna.remove(columna[len(columna) - 1])
columna.append('Decision')
df.columns = columna

#models =['ID3', 'C4.5', 'CART', 'CHAID']
models =[ 'C4.5']
from chefboost import Chefboost as chef
for algorithm in models:
	print("************************",algorithm)
	config = {'algorithm': algorithm}
	model = chef.fit(df.copy(), config)
	count=0
	for index, instance in df.iterrows():
		prediction = chef.predict(model, instance)
		actual = instance['Decision']
		print('actual',actual,'vs','prediction',prediction, 'match',actual==prediction)
		if(actual == prediction):
			count += 1
	print(algorithm, 'match', round(count/df.shape[0]*100,2))

'''
feature = ['Sunny', 'Hot', 'High', 'Weak']
#feature = ['Overcast','Cool','High','Weak']
prediction = chef.predict(model, feature)
print("['Sunny', 'Hot', 'High', 'Weak']",prediction)

feature = ['Overcast', 'Hot', 'High', 'Weak']
moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
tree = chef.restoreTree(moduleName)
prediction = tree.findDecision(feature)

print("['Overcast', 'Hot', 'High', 'Weak']",prediction)

chef.save_model(model, "model_de_iris_73.pkl")

feature = ['Rain', 'Hot', 'High', 'Weak']
model = chef.load_model("model_de_iris_73.pkl")
prediction = chef.predict(model, feature)
print(feature,prediction)
'''