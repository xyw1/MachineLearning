"""
Created on Oct 7, 2018
@author: WANG Ruohan
@email: ruohawang2-c@ad.cityu.edu.hk
"""


import numpy as np
from numpy.dual import inv
import random
import logger
from logger import *
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class LeastSquare():

	def train(self,phi_x,y):
		self.theta_pred = inv(phi_x.dot(phi_x.T)).dot(phi_x).dot(y)
		return self.theta_pred
		
	def predict(self,test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred


class RegularizedLeastSquare():
	
	def train(self,phi_x,y):
		self.theta_pred = inv(phi_x.dot(phi_x.T)+1).dot(phi_x).dot(y)
		return self.theta_pred
	def predict(self,test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred


def Lasso():
	pass


def RobustRegression():
	pass


def BayesianRegression():
	pass

	
def cal_mae(true_value,predicted_value):
	mae = np.square(true_value-predicted_value).mean()
	return mae
	
	
def cal_mse(true_value,predicted_value):
	mse = np.abs(true_value-predicted_value).mean()
	return mse
	
	
def feature_transformation_1(x_matrix):#
	order = 5
	x = x_matrix
	phi_x = np.power(x,0)
	for i in range(1,order+1):
		phi_x = np.c_[phi_x,np.power(x,i)]
	phi_x = phi_x.T
	return phi_x
	

def feature_transformation_2(train_x):
	return train_x
	
	
def load_data(txt):
	arry = np.loadtxt(txt)
	return arry
	

def train(training_data_x_file,training_data_y_file,feature_transformation_func):
	x = load_data(training_data_x_file)
	y = load_data(training_data_y_file)
	y = to_col(y)
	phi_x = feature_transformation_func(x)
	
	global LS
	global RLS
	LS = LeastSquare()
	LS.train(phi_x,y)
	
	RLS = RegularizedLeastSquare()
	RLS.train(phi_x,y)
	
	print LS.theta_pred
	#print RLS.theta_pred
	#lasso = Lasso()
	#RR = RobustRegression()
	#BR = BayesianRegression()
	
 
def predict(test_data_x_file,feature_transformation_func):
	test_x = load_data(test_data_x_file)
	test_phi_x = feature_transformation_func(test_x)
	print test_phi_x
	LS.predict(test_phi_x)
	RLS.predict(test_phi_x)
	print LS.f_pred
	#print RLS.f_pred

def to_col(arry):
	return arry.reshape(arry.shape[0],1)
	
def draw_figure(arry_x,arry_y,title):
	x = to_col(arry_x)
	y = to_col(arry_y)
	ar = np.c_[x,y]
	df = pd.DataFrame(ar)
	df.plot(x=0,y=1)
	plt.savefig(title)
	
	
def polynonimal():
	global sampx_txt,sampy_txt,poly_x,poly_y
	sampx_txt = 'PA-1-data-text/polydata_data_sampx.txt'
	sampy_txt = 'PA-1-data-text/polydata_data_sampy.txt'
	poly_x = 'PA-1-data-text/polydata_data_polyx.txt'
	poly_y = 'PA-1-data-text/polydata_data_polyy.txt'
	y = load_data(poly_y)
	feature_transformation = feature_transformation_1
	train(sampx_txt,sampy_txt,feature_transformation)
	predict(poly_x,feature_transformation)
	draw_figure(LS.f_pred,y,'test')
	pass

def count_people():
	global train_x,train_y,test_x,test_y
	train_x = 'PA-1-data-text/count_data_trainx.txt'
	train_y = 'PA-1-data-text/count_data_trainy.txt'
	test_x = 'PA-1-data-text/count_data_testx.txt'
	test_y = 'PA-1-data-text/count_data_testy.txt'
	pass
	


def main():
	polynonimal()	
	
	
if __name__ == '__main__':
	main()
'''	
def add_outlier(sampy):
	...
	return edited_sampy
 
 
def down_sample(arry,fraction):
	return subset
'''

