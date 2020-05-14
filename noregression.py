#regression
import pandas as pd
import numpy as np

df = pd.read_csv("/home/multipodo/Escritorio/cheef/play.csv")
columna = [lista for lista in df.columns]
columna.remove(columna[len(columna) - 1])
columna.append('Decision')
df.columns = columna

#ID3, C4.5, CART, CHAID or Regression
from chefboost import Chefboost as chef
config = {'algorithm': 'ID3'}
model = chef.fit(df, config)

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