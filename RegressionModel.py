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
#import ggplot as gp
from cvxopt import matrix,solvers
plt.switch_backend('agg')


class LeastSquare():

	def train(self,phi_x,y):
		self.theta_pred = inv(phi_x.dot(phi_x.T)).dot(phi_x).dot(y)
		return self.theta_pred
		
	def predict(self,test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred


class RegularizedLeastSquare():
	
	def train(self,phi_x,y,lambda_I):
		self.theta_pred = inv(phi_x.dot(phi_x.T)+lambda_I).dot(phi_x).dot(y)
		return self.theta_pred
	def predict(self,test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred


class Lasso():
	def train(self,x,y,l):
		h_raw = x.dot(x.T)
		top = np.column_stack((h_raw, -h_raw))
		bottom = np.column_stack((-h_raw, h_raw))
		P = np.row_stack((top, bottom))
		y_raw = x.dot(y)
		q_raw = np.vstack((y_raw, -y_raw))
		q_raw.ravel()
		q_raw.shape = (2*x.shape[0],)
		q = l*np.array([1 for i in range(2*x.shape[0])]) - q_raw
		G = -1*np.eye(2*x.shape[0])
		h =  np.zeros((2*x.shape[0],1))
		logging.debug( 'P:%r', P.shape)
		logging.debug( 'q:%r', q.shape)
		logging.debug( 'G:%r', G.shape)
		logging.debug( 'h:%r', h.shape)
		sol = solvers.qp(matrix(P, tc='d'),matrix(q, tc='d'),matrix(G, tc='d'),matrix(h, tc='d'))
		combine = np.array(sol['x'])
		combine.ravel()
		combine.shape = (2, x.shape[0])
		self.theta_pred =  (combine[0]-combine[1]).T
		return self.theta_pred
	def predict(self,test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred


class RobustRegression():
	def train(self,x,y):
		D = x.shape[0]
		n = x.shape[1]
		zero_d = [0 for i in range(D)]
		one_n = [0 for i in range(n)]
		c = matrix(np.array(zero_d+one_n), tc='d')
		top = np.column_stack((-x.T, -np.identity(n)))
		bottom = np.column_stack((x.T, -np.identity(n)))
		A = matrix(np.row_stack((top, bottom)), tc='d')
		raw_y = np.row_stack((-y, y))
		raw_y.ravel()
		raw_y.shape = (2*n,)
		b = matrix(raw_y, tc='d')
		sol = solvers.lp(c, A, b)
		combine = np.array(sol['x'])
		combine.ravel()
		self.theta_pred =  combine[0:D]
		return self.theta_pred
	def predict(self, test_phi_x):
		self.f_pred = test_phi_x.T.dot(self.theta_pred)
		return self.f_pred

		
def BayesianRegression():
	pass

	
def cal_mae(true_value,predicted_value):
	mae = np.square(true_value-predicted_value).mean()
	return mae
	
	
def cal_mse(true_value,predicted_value):
	mse = np.abs(true_value-predicted_value).mean()
	return mse
	
	
def feature_transformation_1(x_matrix):#
	
	x = x_matrix
	phi_x = np.power(x,0)
	for i in range(1,order+1):
		phi_x = np.c_[phi_x,np.power(x,i)]
	phi_x = phi_x.T
	return phi_x
	

def feature_transformation_2(x_matrix):
	return x_matrix
	

def feature_transformation_3(x_matrix):
	#in this case the x_matrix is 9*N matrix
	return np.r_[x_matrix,np.power(x_matrix,2)]

	
def feature_transformation_4(x_matrix):
	#in this case the x_matrix is 9*N matrix
	return np.r_[np.power(x_matrix,0),x_matrix,np.power(x_matrix,2),np.power(x_matrix,3)]
	
	
def load_data(txt):
	arry = np.loadtxt(txt)
	return arry
	

def train(training_data_x,training_data_y,dimension,feature_transformation_func):

	phi_x = feature_transformation_func(training_data_x)
	y = training_data_y
	global LS
	global RLS
	global RR
	LS = LeastSquare()

	LS.train(phi_x,y)
	
	RLS = RegularizedLeastSquare()
	lambda_ = 1
	I = np.eye(dimension)
	RLS.train(phi_x,y,lambda_*I)
	
	print LS.theta_pred
	print RLS.theta_pred
	#lasso = Lasso()
	RR = RobustRegression()
	RR.train(phi_x,y)
	print RR.theta_pred
	#BR = BayesianRegression()
	
 
def predict(test_data_x,feature_transformation_func):
	
	test_phi_x = feature_transformation_func(test_data_x)
	#print 'phi',test_phi_x
	LS.predict(test_phi_x)
	RLS.predict(test_phi_x)
	RR.predict(test_phi_x)


	
def to_col(arry):
	return arry.reshape(arry.shape[0],1)
	
	
def draw_figure(title,x,**kargs):
	x = to_col(x)
	ar = x
	header = []
	for tag,data in kargs.items():
		header.append(tag)
		data = to_col(data)
		ar = np.c_[ar,data]
	df = pd.DataFrame(ar)
	df.columns = ['x']+header
	df.plot(x='x',y=header)
	plt.savefig(title)
	

def shuffle_data_and_get_subset(data_x,data_y,subset_frac):
	n = data_y.shape[0]
	print n
	index = np.random.permutation(n)
	rand_data_x = data_x[index]
	rand_data_y = data_y[index]
	subset_n = int(n*subset_frac)
	return rand_data_x[:subset_n],rand_data_y[:subset_n]

	
def add_outlier(sampy):
	max_ = max(sampy)
	sampy[0] = 10*max_
	sampy[1] = 10*max_
	sampy[2] = 10*max_
	sampy[3] = 10*max_
	return sampy
 	

def polynonimal():
	global sampx_txt,sampy_txt,poly_x,poly_y
	sampx_txt = 'PA-1-data-text/polydata_data_sampx.txt'
	sampy_txt = 'PA-1-data-text/polydata_data_sampy.txt'
	poly_x = 'PA-1-data-text/polydata_data_polyx.txt'
	poly_y = 'PA-1-data-text/polydata_data_polyy.txt'
	sampx_data = load_data(sampx_txt)
	sampy_data = load_data(sampy_txt)
	x = load_data(poly_x)
	y = load_data(poly_y)
	# part b
	global order
	order = 5
	feature_transformation = feature_transformation_1
	train(sampx_data,sampy_data,order+1,feature_transformation)
	predict(x,feature_transformation)
	draw_figure('5order',x=x,true_value=y,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	print cal_mae(y,LS.f_pred),cal_mse(y,LS.f_pred)
	print cal_mae(y,RLS.f_pred),cal_mse(y,RLS.f_pred)
	print cal_mae(y,RR.f_pred),cal_mse(y,RR.f_pred)
	
	# part c: subset
	for subset_frac in [0.2,0.50,0.75]:
		subset_x,subset_y = shuffle_data_and_get_subset(sampx_data,sampy_data,subset_frac)
		train(subset_x,subset_y,order+1,feature_transformation)
		predict(x,feature_transformation)
		draw_figure('5order%spercent'%int(subset_frac*100),x=x,true_value=y,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	
	# part d: add outlier
	outlier_y = add_outlier(sampy_data)
	train(sampx_data,outlier_y,order+1,feature_transformation)
	predict(x,feature_transformation)
	draw_figure('5order_with_outlier',x=x,true_value=y,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	
	# part e: 10 order
	train(sampx_data,sampy_data,order+1,feature_transformation)
	predict(x,feature_transformation)
	draw_figure('10order',x=x,true_value=y,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
		

	
def count_people():
	global train_x,train_y,test_x,test_y
	train_x = 'PA-1-data-text/count_data_trainx.txt'
	train_y = 'PA-1-data-text/count_data_trainy.txt'
	test_x = 'PA-1-data-text/count_data_testx.txt'
	test_y = 'PA-1-data-text/count_data_testy.txt'
	training_data_x = load_data(train_x)
	training_data_y = load_data(train_y);training_data_y = to_col(training_data_y)
	test_data_x = load_data(test_x)	
	y = load_data(test_y)
	x = np.array(range(1,y.shape[0]+1))
	feature_transformation = feature_transformation_2
	train(training_data_x,training_data_y,9,feature_transformation)
	predict(test_data_x,feature_transformation)
	LS.f_pred = LS.f_pred.round()
	RLS.f_pred = RLS.f_pred.round()
	RR.f_pred = RR.f_pred.round()
	draw_figure('count_peole',x=x,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	print cal_mae(y,LS.f_pred),cal_mse(y,LS.f_pred)
	print cal_mae(y,RLS.f_pred),cal_mse(y,RLS.f_pred)
	print cal_mae(y,RR.f_pred),cal_mse(y,RR.f_pred)
	
	feature_transformation = feature_transformation_3
	train(training_data_x,training_data_y,18,feature_transformation)
	predict(test_data_x,feature_transformation)
	LS.f_pred = LS.f_pred.round()
	RLS.f_pred = RLS.f_pred.round()
	RR.f_pred = RR.f_pred.round()
	draw_figure('count_people_nonLinear_feature_transformation_1',x=x,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	print cal_mae(y,LS.f_pred),cal_mse(y,LS.f_pred)
	print cal_mae(y,RLS.f_pred),cal_mse(y,RLS.f_pred)
	print cal_mae(y,RR.f_pred),cal_mse(y,RR.f_pred)
	
	
	feature_transformation = feature_transformation_4
	train(training_data_x,training_data_y,36,feature_transformation)
	predict(test_data_x,feature_transformation)
	LS.f_pred = LS.f_pred.round()
	RLS.f_pred = RLS.f_pred.round()
	RR.f_pred = RR.f_pred.round()
	draw_figure('count_people_nonLinear_feature_transformation_2',x=x,LS_predicted=LS.f_pred,RLS_predicted=RLS.f_pred,RR_predicted=RR.f_pred)
	print cal_mae(y,LS.f_pred),cal_mse(y,LS.f_pred)
	print cal_mae(y,RLS.f_pred),cal_mse(y,RLS.f_pred)
	print cal_mae(y,RR.f_pred),cal_mse(y,RR.f_pred)

	


def main():
	polynonimal()	
	count_people()
	
	
if __name__ == '__main__':
	main()
	

 


