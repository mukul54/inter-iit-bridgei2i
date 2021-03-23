import os
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
import json

model_brand = pd.read_csv('./model_brands3.csv')

brands = model_brand.brands.unique()
model = model_brnd.model.unique()
mb_lis = list(set(list(brands)+list(model)))

dic = {x : 500 for x in model_brands}

# Serialize data into file:
json.dump(dic, open( "./ekphrasis/ekphrasis/dicts/brand_models.json", 'w' ) )

# Read data from file:
model_brands = json.load( open( "./ekphrasis/ekphrasis/dicts/brand_models.json" ))