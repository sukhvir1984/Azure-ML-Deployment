# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:37:34 2021

@author: agarw
"""
import pandas as pd
import os



data=pd.read_csv("./diabetes.csv")
data=data.loc[data.age<=70]
data=data.loc[data.glucose_concentration!=0]

parser = argparse.ArgumentParser()
parser.add_argument('--datastoarge', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'diabeticsclean.csv')


data.to_csv(output_path)