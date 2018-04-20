# coding: utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import Imputer
import datetime
mpl.rc('figure', figsize=[10,6])

df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')

print(df.head())