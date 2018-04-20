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
#
# mpl.rc('figure', figsize=[10, 6])
#
# df = pd.read_csv('wdbc.data', header=None)
# base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
# 			  'conpoints', 'symmetry', 'fracdim']
# names = ['m' + name for name in base_names]
# names += ['s' + name for name in base_names]
# names += ['e' + name for name in base_names]
# names = ['id', 'class'] + names
# df.columns = names
# df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
# my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')
#
# print(df.head())

class LogisticRegression:
	def __init__(self, X, y, alpha, reg_strength, number_of_iterations):
		self.X = np.insert(X, 0, 1, axis=1)
		self.y = y
		self.alpha = alpha
		self.reg_strength = reg_strength
		self.number_of_iterations = number_of_iterations
		self.theta = np.zeros((self.X.shape[1], 1))
		self.m = len(X)


	def gradient_descent(self):
		if len(self.X) != len(self.y):
			raise ValueError(f'Lengths of X and y do not match. X is {len(self.X)} and y is {len(self.y)}')

		theta = np.zeros((len(self.X) + 1, 1))

		return theta


	# Get new weights
	def update_weights(self):
		h_x = self.get_sig()
		new_weights = np.subtract(self.y, h_x)
		#print(new_weights)
		#print(new_weights.shape)
		#print(self.X.shape)

		new_weights = self.X * new_weights
		#print(new_weights)
		off_set_thetas = 2 * self.reg_strength * self.theta
		#print(off_set_thetas)
		new_weights = np.subtract(new_weights, off_set_thetas.T)
		#print(new_weights)
		new_weights = np.sum(new_weights, axis=0)
		#print(new_weights)

		alpha_over_m = self.alpha / self.m
		new_weights = alpha_over_m * new_weights
		#print(new_weights)

		self.theta = (self.theta.T + new_weights).T
		#print(f'theta{self.theta}')


	# Get result of sigmoid function
	def get_sig(self):
		weighted_values = np.matmul(self.X, self.theta)
		return 1/(1 + np.exp(-weighted_values))


	def get_loss(self):
		# Get sigmoid for each row in X.
		h_x = self.get_sig()
		# Loss function
		loss = self.y * np.log(h_x)
		loss += np.subtract(np.ones(self.y.shape), self.y) * np.log(np.subtract(np.ones(h_x.shape), h_x))
		loss = -np.sum(loss)
		# Calculate complexity.
		complexity = self.reg_strength * np.sum(np.power(self.theta, 2))
		return (loss + complexity)/self.m


lr = LogisticRegression(np.array([[1,3,1], [2,2,5], [3,2,3]]), np.array([[1],[0],[1]]), .2, 4, 100)
#print(lr.get_loss())
for i in range(0,1000):
	print(lr.get_loss())
	lr.update_weights()
	#print(lr.theta)
