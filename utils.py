from os.path import exists, isfile
import sys
import json

import numpy as np
import pandas as pd

END = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
BLACK = '\033[1;30m'
RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[1;34m'
VIOLET = '\033[1;35m'
CYAN = '\033[1;36m'
WHITE = '\033[1;37m'

# --------------------------------------------------------------------------- #
# Functions related to the process of linear regression: prediction, cost,    #
# gradient , fit, ...                                                         #
# --------------------------------------------------------------------------- #

def simple_predict(x:float, theta:np.ndarray) -> float:
	""" Prediction using x and vector coefficients theta.
	Parameters:
		x: (np.float64) x value
		theta: (np.ndarray) vector coefficient

	Return:
		pred: (np.float64) predicted value
	"""
	if theta.shape != (2,1):
		str_xpt = "theta vector is not of correct shape."
		raise Exception(str_xpt)
	
	pred = theta[1] * x + theta[0]
	return pred


def predict(x:np.ndarray, theta:np.ndarray) -> np.ndarray:
	""" Prediction using x and vector coefficients theta.
	Parameters:
		x: (np.float64) x value
		theta: (np.ndarray) vector coefficient

	Return:
		pred: (np.float64) predicted value
	"""
	if theta.shape != (2,1):
		str_xpt = "theta vector is not of correct shape."
		raise Exception(str_xpt)
	
	m = x.shape[0]
	xp = np.hstack((np.ones((m,1)),x))
	pred = np.dot(xp,theta)
	return pred.reshape(-1,1)


def cost_elem(ypred : np.ndarray, y : np.ndarray) -> np.ndarray:
	""" Calculates the residual between each component of the prediction
	and the corresponding true values.
	Parameters:
		ypred: (np.ndarray) vector of predicted values
		y: (np.ndarray) vector true value

	Return:
		cost_elem: (np.ndarray) residuals vector
	"""
	if ypred.ndim != 2 or y.ndim != 2 :
		str_ypredpt = "Incorrect dimension for x or/and y array."
		raise Exception(str_xpt)
	if ypred.shape[1] != 1 or y.shape[1] != 1:
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)
	
	m = ypred.shape[0]
	cost_elem = (1 / m) * np.square(ypred - y)
	return cost_elem


def cost(ypred:np.ndarray, y:np.ndarray) -> np.float64:
	""" Calculates the cost between the vector of prediction and of true value
	Parameters:
		ypred: (np.ndarray) vector of predicted values
		y: (np.ndarray) vector true value

	Return:
		cost: (np.float64) cost between ypred and y
	"""
	if ypred.ndim != 2 or y.ndim != 2 :
		str_xpt = "Incorrect dimension for x or/and y array."
		raise Exception(str_xpt)
	if ypred.shape[1] != 1 or y.shape[1] != 1:
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)
	
	cost = 0.5 * np.sum(cost_elem(ypred,y))
	return cost


def gradient(x:np.ndarray, y:np.ndarray, theta:np.ndarray) -> np.ndarray:
	"""Calculates the gradient of the cost function along the theta directions
	Parameters:
		x: (np.ndarray) predicted values vector
		y: (np.ndarray) true values vector
		theta: (np.ndarray)  hypothesis's coefficients vector

	Return:
		grad: (np.ndarray) gradient along theta directions
	"""
	if x.ndim != 2 or y.ndim != 2 :
		str_xpt = "Incorrect dimension for x or/and y array."
		raise Exception(str_xpt)
	if x.shape[1] != 1 or y.shape[1] != 1:
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)
	if theta.ndim != 2 or theta.shape != (2,1):
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)

	m = x.shape[0]
	xp = np.hstack((np.ones((m,1)),x))
	grad = (1 / m) * np.dot(xp.T, (predict(x, theta) - y))
	return grad


def fit(x:np.ndarray, y:np.ndarray, theta:np.ndarray, alpha:float=2e-2, max_iter:int=10000, delta=None) -> np.ndarray:
	""" Performs the gradient descent and update the theta vector
	Parameters:
		x: (np.ndarray) vector of predicted values
		y: (np.ndarray) vector true value
		theta: (float) coefficients vector (model parameters)
		alpha: (float) learning rate
		max_iter: (int) number of iteration of the gradient descent

	Return:
		theta: (np.ndarray) hypothesis's coefficients vector
	"""    
	if x.ndim != 2 or y.ndim != 2 :
		str_xpt = "Incorrect dimension for x or/and y array."
		raise Exception(str_xpt)
	if x.shape[1] != 1 or y.shape[1] != 1:
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)
	if theta.ndim != 2 or theta.shape != (2,1):
		str_xpt = "Incorrect shape for x or/and y array."
		raise Exception(str_xpt)
	if isinstance(delta, (float)) and delta > 0 and delta < 0.1:
		for i in range(max_iter):
			diff = gradient(x,y,theta)
			if all(diff < delta):
				print(f"Stop at iteration: {i} (gradient: {diff.reshape(-1,)})")
				break
			theta = theta - alpha * diff
	else:
		for i in range(max_iter):
			theta = theta - alpha * gradient(x,y,theta)
	return theta


def normal_equation(x:np.ndarray, y:np.ndarray) -> np.ndarray:
	""" Performs the normal equation method to calculates
	the best value for theta vector
	Arguments:
		x: (np.ndarray) vector of predicted values
		y: (np.ndarray) vector true value
	Return:
		theta: (np.ndarray) best hypothesis's coefficients vector
	Ressource:
		- explanation of the method:
		https://en.wikipedia.org/wiki/Ordinary_least_squares
		- explanation on why to check invertibility:

		- Gram matrix:
		https://en.wikipedia.org/wiki/Gram_matrix
	"""
	xp = np.hstack((np.ones((x.shape[0],1)),x))
	Gram_matrix = np.dot(xp.T, xp)
	det_Gram = np.linalg.det(Gram_matrix)
	if det_Gram == 0:
		print(RED + "xp.T*xp is not invertible, due to associated Gram matrix having a null determinant." + END)
		sys.exit()
	theta = np.dot(np.linalg.inv(np.dot(xp.T,xp)), np.dot(xp.T,y))
	return theta


def least_square_linear_reg_derivation(x:np.ndarray, y:np.ndarray) -> np.ndarray:
	""" Performs the least square linear regression derivation method
	to get the best value for theta vector
	Arguments:
		x: (np.ndarray) vector of predicted values
		y: (np.ndarray) vector true value
	Return:
		theta: (np.ndarray) best hypothesis's coefficients vector
	Ressource:
		- article explaining the method:
		http://nitro.biosci.arizona.edu/courses/EEB581-2006/handouts/bivdistrib.pdf
	"""
	cov_matrix = np.cov(x.reshape(-1,), y.reshape(-1,))
	theta1 = cov_matrix[0][1] / cov_matrix[0][0]
	theta0 = np.mean(y) - theta1 * np.mean(x) 
	return np.array([[theta0], [theta1]])


# --------------------------------------------------------------------------- #
# Function for JSON or CSV file interaction                                   #
# --------------------------------------------------------------------------- #
def open_read_coeff() -> np.ndarray:
	""" Reads if possible the JSON file containing the model/hypothesis's
	coefficients. If the file does not exist or is not a correct JSON file
	theta vector is set to numpy.ndarray[([0.0], [0.0]]).
	Return:
		theta: (np.ndarray) hypothesis's coefficients vector
	"""
	if not isfile('coefficients.json'):
		print('JSON file coefficients.json does not exist.')
		print('theta vector is initiated to [0.0  0.0]')
		theta = np.array([[0.0], [0.0]])
	else:
		with open("coefficients.json", 'r') as open_file:
			try:
				data = json.load(open_file)
			except:
				print('JSON file coefficients.json seems to be corrupted or empty.')
				print('theta vector is initiated to [0.0  0.0]')
				data = {"theta0" : 0.0, "theta1" : 0.0}
			
			l = len(data.keys()) 
			if  l != 2:
				print("Missing or extra keys in JSON file.")
				print('theta vector is initiated to [0.0  0.0]')
				data = {"theta0" : 0.0, "theta1" : 0.0}
			elif any([key not in ["theta0", "theta1"] for key in data.keys()]):
				print("Unexpected key(s) for theta.")
				print('theta vector is initiated to [0.0  0.0]')
				data = {"theta0" : 0.0, "theta1" : 0.0}
			else:
				print("All keys in coefficients.json are valid.")
		theta = np.array([np.float64(data["theta0"]), np.float64(data["theta1"])]).reshape(-1,1)
	return theta

def open_write_coeff(theta:np.ndarray):
	""" Writes if possible the theta in JSON file containing.
	If the file does not exist then it is created.
	"""
	dct_theta = {"theta0":str(theta[0]).strip('[]'),
					"theta1":str(theta[1]).strip('[]')}
	if not isfile('coefficients.json'):
		print('Creation of JSON file coefficients.json.')
	with open("coefficients.json", 'w') as open_file:
			try:
				data = json.dump(dct_theta, open_file)
				print("theta components have been written in coefficients.json.")
			except:
				print('Error during writting of theta in JSON file.')

def open_read_data() -> pd.DataFrame:
	""" Reads if possible the CSV file containing the data.
	If the file does not exist or is not a correct CSV file
	the program quit.
	Return:
		df: (pd.DataFrame) dataframe containing the raw dataset.
	"""
	if not exists('./data/data.csv'):
		str_err = "Path data/data.csv does not exist."
		print(str_err)
		sys.exit()

	print("Reading data file ...")
	try:
		df = pd.read_csv("data/data.csv", index_col=False)
	except pd.errors.EmptyDataError as err:
		s_expt = "No columns to parse from file."
		print(s_expt)
		sys.exit()
	except:
		s_expt = ("Unexpected error.")
		print(s_expt)
		sys.exit()
	
	# -- Specific verification of the dataset for the project ft_linear_regression
	b_dtypes = [pd.api.types.is_numeric_dtype(df.iloc[:,0]), pd.api.types.is_numeric_dtype(df.iloc[:,1])]
	if (df.shape != (24,2)) or not (df.columns[0] == "km" and df.columns[1] == "price") \
		or (not all(b_dtypes)) or any(df.isnull().values.reshape(-1,)):
		print("Dataset not well formated.")
		sys.exit()
	df.sort_values(by='km', inplace = True)
	return df

# --------------------------------------------------------------------------- #
# Function to display the usage of fit                                        #
# --------------------------------------------------------------------------- #
def print_usage():
	""" Display the usage of the program fit.py
	"""
	str_usage = GREEN + "Usage:\n" + END
	str_usage += f"  python fit.py {BLUE}--graphic=...{END} {YELLOW}--method=...{END}\n"
	str_usage += GREEN + "Args:\n" + END
	str_usage += BOLD + "  --graphic=[console/static/dynamic]\n" + END
	str_usage += f"     * {BLUE}console{END}: (default) no graphic, only results in terminal.\n"
	str_usage += f"     * {BLUE}static{END}:  display 2 plots: the raw data with the model curve\n"
	str_usage += "                and the cost function with respect to the iteration.\n"
	str_usage += f"     * {BLUE}dynamic{END}: diplay 3 plots: data with model curve.\n"
	str_usage += "                plus the cost function with respect to the iteration.\n"
	str_usage += "                plus the contour plot of cost function and current cost value (red dot).\n"
	str_usage += BOLD + "  --method=[gradient-descent/normal-equation/covariance]:\n" + END
	str_usage += f"     * {YELLOW}gradient-descent{END}: (default) method of the gradient descent.\n"
	str_usage += f"     * {YELLOW}normal-equation{END}:  method based on the normal equation.\n"
	str_usage += f"     * {YELLOW}covariance{END}:       least square linear regression derivation method.\n"
	print(str_usage)