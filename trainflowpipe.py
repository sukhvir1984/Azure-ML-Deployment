# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:37:34 2021

@author: agarw
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
import pandas as pd
import joblib
import os


parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder
output_path = os.path.join(output_folder, 'diabeticsclean.csv')

data=pd.read_csv(output_path)
print(data.columns)

data=data

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

trainx,testx,trainy,testy=train_test_split(X, y, random_state=101, test_size=0.2)

lg=RandomForestClassifier()
print(lg)
lg.fit(trainx,trainy)

y_pred=lg.predict(testx)

classification_report(testy,y_pred)

confusion_matrix(testy,y_pred)
acc=accuracy_score(testy,y_pred)

print(accuracy_score(testy,y_pred))
joblib.dump(lg, "diabeticmodeltest.pkl")

