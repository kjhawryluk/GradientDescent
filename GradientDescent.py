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
mpl.rc('figure', figsize=[10, 6])

df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
			  'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
non_class_columns = [c for c in df.columns if c != 'id' and c != 'class']
# Normalize Columns
col_max_values = df[non_class_columns].max()
col_min_values = df[non_class_columns].min()
col_avg_values = df[non_class_columns].mean()
df[non_class_columns] = (df[non_class_columns] - col_avg_values)/(col_max_values - col_min_values)
# Recode class
df['class'] = df['class'].map({'M': 0, 'B': 1})
#
#
#
# print(df.head())

class HawrylukLogisticRegression:
	def __init__(self, X, y, alpha, reg_strength, number_of_iterations, convergence_sensitivity=.000001):
		self.X = np.insert(X, 0, 1, axis=1)
		self.y = y
		self.alpha = alpha
		self.reg_strength = reg_strength
		self.number_of_iterations = number_of_iterations
		self.m = len(X)
		self.theta = np.zeros((self.X.shape[1], 1))
		self.loss_by_iteration = []
		self.convergence_sensitivity = convergence_sensitivity


	def gradient_descent(self):
		if len(self.X) != len(self.y):
			raise ValueError(f'Lengths of X and y do not match. X is {len(self.X)} and y is {len(self.y)}')

		# Calculate gradient descent. Store list of loss and iteration to plot.
		i = 0
		while i < self.number_of_iterations and not self.has_converged():
			i += 1
			self.update_weights()
			self.loss_by_iteration.append(self.get_loss())

		return self.theta, self.loss_by_iteration

	# Check if the difference between the iteration of the last iteration and that of this one is less than the
	# convergence sensitivity.
	def has_converged(self):
		if len(self.loss_by_iteration) < 2:
			return False
		return abs((self.loss_by_iteration[-1] - self.loss_by_iteration[-2])) < self.convergence_sensitivity

	# Get new weights
	def update_weights(self):
		# Subtract sigmoid value of rows from their ys and multiply by the rows.
		h_x = self.get_sig()
		new_weights = np.subtract(self.y, h_x)
		new_weights = self.X * new_weights

		# Calculate 2 * lambda * theta
		off_set_thetas = 2 * self.reg_strength * self.theta

		# Subtract those two
		new_weights = np.subtract(new_weights, off_set_thetas.T)

		# Sum over all rows to flatten into column coefficients.
		new_weights = np.sum(new_weights, axis=0)

		# Apply weight
		alpha_over_m = self.alpha / self.m
		new_weights = alpha_over_m * new_weights

		# Update weights
		self.theta = (self.theta.T + new_weights).T


	# Get result of sigmoid function
	def get_sig(self):
		weighted_values = np.matmul(self.X, self.theta)
		return 1/(1 + np.exp(-weighted_values))

	# This applies a floor so that a log of 0 or below never happens.
	def subtract_h_x_from_one(self, h_x):
		difference = np.subtract(np.ones(h_x.shape), h_x)
		difference[difference <= 0] = .0001
		return difference

	def get_loss(self):
		# Get sigmoid for each row in X.
		h_x = self.get_sig()
		# Loss function
		loss = self.y * np.log(h_x)
		one_minus_h_x = self.subtract_h_x_from_one(h_x)
		loss += np.subtract(np.ones(self.y.shape), self.y) * np.log(one_minus_h_x)
		loss = -np.sum(loss)
		# Calculate complexity.
		complexity = self.reg_strength * np.sum(np.power(self.theta, 2))
		return (loss + complexity)/self.m


def get_class_column_as_2d_matrix(df, col_name):
	return df[col_name].as_matrix().reshape(1, len(df[col_name])).T


def plot_iterations_and_loss(loss_by_iteration):
	loss_values = loss_by_iteration
	iteration_value = range(0, len(loss_values))
	title = 'Loss by Iteration Number'
	plt.clf()
	plt.title(title)
	plt.ylabel("L2 Loss")

	# Plot param range on x axis.
	plt.plot(iteration_value, loss_values, label="L2 Loss", color="navy")

	plt.legend(loc="best")
	plt.savefig(title + ".pdf")


def compare_gradient_descent():
	lr = HawrylukLogisticRegression(df[non_class_columns].values, get_class_column_as_2d_matrix(df, 'class'), 10, .0009, 200)
	theta, loss_by_iteration = lr.gradient_descent()
	plot_iterations_and_loss(loss_by_iteration)
	print(f'My loss after {len(loss_by_iteration)} iterations: {np.round(loss_by_iteration[-1],3)}')
	clf = LogisticRegression()
	clf.fit(df[non_class_columns], df['class'])
	print(f'My Intercept: {np.round(theta.T[0,0], 3)}, SciKit\'s Intercept: {np.round(clf.intercept_[0],3)}.')

	x = 1
	for mine, theirs in zip(theta.T[0,1:], clf.coef_[0]):
		print(f'My Coefficient {x}: {np.round(mine, 3)}, SciKit\'s Coefficient {x}: {np.round(theirs,3)}.')
		x += 1

def regress_mradius_and_mtexture():
	c1 = 'mradius'
	c2 = 'mtexture'
	title = 'Breast Cancer Classification Predictions'
	# Set up plot and plot samples.
	my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')
	plt.clf()
	plt.scatter(df[c1], df[c2], c=df['class'], cmap=my_color_map)
	plt.title(title)
	plt.xlabel(c1)
	plt.ylabel(c2)

	# Add terms up to degree 3
	features = df[[c1, c2]]
	polynomial_combinations = sklearn.preprocessing.PolynomialFeatures(3)
	poly_columns = polynomial_combinations.fit_transform(features)

	# Drop intercept column because this is done by sci kit learn
	poly_columns = np.delete(poly_columns, 0, 1)

	# Fit model
	clf = LogisticRegression()
	clf.fit(poly_columns, df['class'])

	# Get all combos of mradius and mtexture
	x = np.linspace(df[c1].min(), df[c1].max(), 1000)
	y = np.linspace(df[c2].min(), df[c2].max(), 1000)
	xx, yy = np.meshgrid(x, y)
	flatten_xx_and_yy = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))

	# Add polynomial variations to combos.
	prediction_combos = polynomial_combinations.fit_transform(flatten_xx_and_yy)
	prediction_combos = np.delete(prediction_combos, 0, 1)

	# Make prediction
	predictions = clf.predict(prediction_combos)
	predictions = predictions.reshape(xx.shape)

	# Plot contour according to prediction.
	plt.contour(xx, yy, predictions, [0.0])
	plt.savefig(title + ".pdf")


if __name__ == "__main__":
	compare_gradient_descent()
	regress_mradius_and_mtexture()