# Libraries related to data manipulation
import numpy as np

#
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
n_frames, interval = 100, 5

# Import of the functions in the local module utils
from utils import *

# Constante for color printing
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
# Functions definition related to the animated visualization                  #
# --------------------------------------------------------------------------- #
def animate_visu() -> np.ndarray:
	""" Core function for the animated vizualisation.
	The function defines all the x/y_labels, the titles.
	"""

	plt.style.use('seaborn-pastel')
	global line, line_cost, pt_curr_cost, axes
	
	# Formatting figure and axes:
	fig = plt.figure(figsize=(12,10))
	gs = GridSpec(2, 2, figure=fig)
	axes = []
	axes.append(fig.add_subplot(gs[:, 0]))
	axes.append(fig.add_subplot(gs[0, 1]))
	axes.append(fig.add_subplot(gs[1, 1]))
	axes[0].set_xlabel("standardized mileage")
	axes[0].set_ylabel("standardized price")
	axes[1].set_xlabel("i: iteration")
	axes[1].set_ylabel(r"$\mathcal{L}_{\theta_0,\theta_1}$")
	axes[2].set_xlabel(r"$\theta_0$")
	axes[2].set_ylabel(r"$\theta_1$")
	axes[0].grid()
	axes[1].grid()
	axes[0].set_title(r'Price(km) vs predicted Price(arbitrary unit)')
	axes[1].set_title(r'$\mathcal{L}_{\theta_0,\theta_1}$ vs iteration')
	axes[2].set_title(r'Contours Plot $\mathcal{L}(\theta_0, \theta_1)$')
	
	# Initialisation of the Line2D object for the differents Axes objects
	raw_data = axes[0].scatter(x,y, color='grey', s= 20)
	line, = axes[0].plot([], [])
	line_cost, = axes[1].plot([], [], color="red", lw=1.2, marker='o', ms=1.5)
	pt_curr_cost, = axes[2].plot([], [], linestyle = '', marker='.', color="red")
	
	# Contour plot construction
	delta = 1e-2
	theta0 = np.arange(-1, 1, 0.5*delta).reshape(-1,1)
	theta1 = np.arange(-2.75, 1.25, delta).reshape(-1,1)
	Theta0, Theta1 = np.meshgrid(theta0, theta1)
	
	J_std = [(Theta1 * xi + Theta0 - yi)**2 for xi,yi in zip(x, y)]
	J_std_arr = np.mean(np.array(J_std), axis = 0)

	cp = axes[2].contourf(Theta0, Theta1, J_std_arr)
	cs = axes[2].contour(Theta0, Theta1, J_std_arr, colors="black")
	axes[2].clabel(cs, inline=True, fontsize=10, colors="black")
	fig.colorbar(cp) 

	anim_data = FuncAnimation(fig, animate_data, frames=n_frames, interval=interval, repeat=False, cache_frame_data = False)
	#anim_cost = FuncAnimation(fig, animate_cost, frames=n_frames, interval=interval, repeat=False, cache_frame_data = False)
	#anim_cost_curve = FuncAnimation(fig, animate_cost_theta, frames=n_frames, interval=interval, repeat=False, cache_frame_data = False)
	plt.waitforbuttonpress()
	plt.show()


def animate_data(i):
	""" Function for the animation of the 1st axe (representation of the data)
	"""
	global theta, x,  cost_x, cost_y, lst_theta
	n_cycle = int(2000 / n_frames)
	theta = fit(x, y, theta, max_iter = n_cycle)
	line.set_data(x, predict(x, theta))

	coord_x = i * int(2000 / n_frames)
	cost_x.append(coord_x)
	cost_y.append(cost(predict(x, theta), y))
	axes[1].set_xlim(-100, cost_x[-1] + 500)
	axes[1].set_ylim(0, max(cost_y) * 1.1)
	line_cost.set_data(cost_x, cost_y)

	if len(lst_theta) == 0:
		lst_theta = np.append(lst_theta, theta)
		pt_curr_cost.set_data(lst_theta[0], lst_theta[1])
	else:
		lst_theta = np.vstack((lst_theta, theta.T))
		pt_curr_cost.set_data(lst_theta[:,0], lst_theta[:,1])

	return line, line_cost, pt_curr_cost,

# --------------------------------------------------------------------------- #
# Functions definition related to the static visualization or console output  #
# --------------------------------------------------------------------------- #
def static_visu(x:np.ndarray, y:np.ndarray, theta:np.array, method:str) -> np.ndarray:
	""" Function managing the graphical representation in the context of the
	static mode.
	Parameters:
	-----------
		x [np.array]: examples/input vector.
		y [np.array]: target values vector.
		theta [np.array]: array containing 2 components being the coefficient of the model.
		method [str]: ['gradient-descent'/'normal-equation'/'covariance'] the method used.
	Return:
		theta [np.array]: value of coefficient vector after training.
	-------

	"""
	lst_theta = [theta]
	lst_cost = [cost(predict(x, theta), y)]
	if method == 'gradient-descent':
		alpha = 1e-2
		max_iter = 10000
		delta = None
		nb_step = 20
		method_str = "=" * 30 + "\n" + \
					"Method: Gradient Descent\n" + \
					"Parameters:\n" + \
					"\talpha(learning rate)  =".ljust(30) + f"{alpha}\n" + \
					"\tnumber of iterations  =".ljust(30) + f"{max_iter}\n" + \
					"\tconvergence criterion =".ljust(30) + f"{delta}\n" + \
					"\tnb_step               = ".ljust(30) + f"{nb_step}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		
		count = 0
		lst_theta = [theta]
		lst_cost = [cost(predict(x, theta), y)]
		while count < max_iter:
			theta = fit(x, y, theta, alpha, nb_step)
			count += nb_step
			lst_theta.append(theta)
			lst_cost.append(cost(predict(x, theta), y))
	if method == 'normal-equation':
		xp = np.hstack((np.ones((len(x),1)),x))
		Gram_matrix = np.dot(xp.T,xp)
		method_str = "=" * 30 + "\n" + \
					"Method: Normal Equation\n" + \
					"Parameters:\n" + \
					f"Gram matrix =\n{Gram_matrix}\n" + \
					f"(det(Gram_matrix) = {np.linalg.det(Gram_matrix):.3e}" + \
					" if non null, it means the column vectors" + \
					" are linearly independent.)\n\n" + \
					f"Moment matrix =\n{np.dot(xp.T,y)}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		nb_step = 1
		max_iter = 1
		theta = normal_equation(x,y)
		lst_theta.append(theta)
		lst_cost.append(cost(predict(x, theta), y))
	if method == 'covariance':
		cov_matrix= np.cov(x.reshape(-1,),y.reshape(-1,))
		method_str = "=" * 30 + "\n" + \
					"Method: Least Square Linear Regression derivation\n" + \
					"Parameters:\n" + \
					f"\tcovariance matrix =\n{cov_matrix}\n" + \
					f"\tmean(x) = {np.mean(x)}\n" + \
					f"\tmean(y) = {np.mean(y)}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		nb_step = 1
		max_iter = 1
		theta = least_square_linear_reg_derivation(x, y)
		lst_theta.append(theta)
		lst_cost.append(cost(predict(x, theta), y))
	
	fig, axes = plt.subplots(1,2, figsize=(12,8))
	if method == "gradient-descent":
		axes[0].set_xlabel("standardized mileage")
		axes[0].set_ylabel("standardized price")
	else:
		axes[0].set_xlabel("mileage (km)")
		axes[0].set_ylabel("price")
	axes[1].set_xlabel("i: iteration")
	axes[1].set_ylabel(r"$\mathcal{L}_{\theta_0,\theta_1}$")
	axes[0].set_title(r'Price(km) vs predicted Price(arbitrary unit)')
	axes[1].set_title(r'$\mathcal{L}_{\theta_0,\theta_1}$ vs iteration', pad=20)
	axes[0].grid()
	axes[1].grid()
	axes[0].scatter(x,y, color='grey', s= 20)
	axes[0].plot(x,predict(x, theta))
	axes[1].plot(np.arange(0,max_iter + nb_step, nb_step), lst_cost,
						   color="red", ls='-', marker='o', ms=2, lw=1.2)
	plt.show()
	return theta


def console_visu(x:np.ndarray, y:np.ndarray, theta:np.ndarray, method:str) -> np.ndarray:
	""" Function manages the display in the console plus perform the fit process.
	Parameters:
		* x [np.ndarray]: input data/examples, here it should be a (m,1) numpy array.
		* y [np.ndarray]: output data/target, here it should be a (m,1) numpy array.
		* theta []:
		* method [str=gradient-descent|"normal-equation"|"covariance"]: method to use.
	"""
	if method == 'gradient-descent':
		alpha = 1e-2
		max_iter = 10000
		delta = None
		method_str = "=" * 30 + "\n" + \
					"Method: Gradient Descent\n" + \
					"Parameters:\n" + \
					f"\talpha(learning rate)  = {alpha}\n" + \
					f"\tnumber of iterations  = {max_iter}\n" + \
					f"\tconvergence criterion = {delta}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		theta = fit(x, y, theta, alpha, max_iter, delta)
	if method == 'normal-equation':
		xp = np.hstack((np.ones((len(x),1)),x))
		method_str = "=" * 30 + "\n" + \
					"Method: Normal Equation\n" + \
					"Parameters:\n" + \
					f"Normal matrix =\n{np.dot(xp.T,xp)}\n\n" + \
					f"Moment matrix =\n{np.dot(xp.T,y)}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		theta = normal_equation(x,y)
	if method == 'covariance':
		cov_matrix= np.cov(np.stack((x.reshape(-1,),y.reshape(-1,)),axis=0))
		method_str = "=" * 30 + "\n" + \
					"Method: Least Square Linear Regression derivation\n" + \
					"Parameters:\n" + \
					f"\tcovariance matrix =\n{cov_matrix}\n" + \
					f"\tmean(x) = {np.mean(x)}\n" + \
					f"\tmean(y) = {np.mean(y)}\n" + \
					"=" * 30 + "\n"
		print(method_str)
		theta = least_square_linear_reg_derivation(x, y)
	return theta


# ########################################################################### #
# ___________________________________ MAIN __________________________________ #
# ########################################################################### #
if __name__ == "__main__":
	# -- Parsing of arguments:
	b_visu = b_dynamic = b_static = b_console = b_descent_grad = \
		b_normal_equation = b_covariance = b_method = False
	lst_possible_args = ["--graphic=console",
						 "--graphic=static",
						 "--graphic=dynamic",
						 "--method=gradient-descent",
						 "--method=normal-equation",
						 "--method=covariance"]

	# -- Parsing of the parameters
	argv = sys.argv[1:]
	if len(argv) == 1 and (argv[0] in ["-h", "--help", "--usage"]):
		print_usage()
		sys.exit()
	for arg in argv:
		if (arg == "--graphic=console") and (b_visu == False):
			b_visu = True
			b_console = True
		elif (arg == "--graphic=static") and (b_visu == False):
			b_visu = True
			b_static = True
		elif (arg == "--graphic=dynamic") and (b_visu == False):
			b_visu = True
			b_dynamic = True
		elif (arg == "--method=gradient-descent") and (b_method == False):
			b_method = True
			b_descent_grad = True
			method = arg.split('=')[1]
		elif (arg == "--method=normal-equation") and (b_method == False):
			b_method = True
			b_normal_equation = True
			method = arg.split('=')[1]
		elif (arg == "--method=covariance") and (b_method == False):
			b_method = True
			b_covariance = True
			method = arg.split('=')[1]
		else:
			if arg not in lst_possible_args:
				print("Invalid argument.")
			else:
				print("method or graphic argument cannot be define more than once.")
			sys.exit()
	
	# -- Correction of the 'mode' if dynamic visualization and covariance
	# -- /normal equation are enabled (covariance and normal equation
	# -- are one step resolution methods)
	if b_dynamic and (b_normal_equation or b_covariance):
		print("Dynamic visualization is compatible only with gradient descent method.")
		print("Static visualization is enforced.")
		b_dynamic = False
		b_static = True

	# -- In case of no arguments default behavior is set:
	if b_method == False:
		b_descent_grad = True
		method = "gradient-descent"
	if b_visu == False:
		b_console = True

	# -- Read of the dataset and the linear hypothesis coefficients
	# -- If coefficients.json file does not existed, theta are set
	# -- to np.array([0., 0.])
	df = open_read_data()
	theta = open_read_coeff()
	
	# -- Initialisation and standardization of x and y vectors:
	x = df['km'].values.reshape(-1,1)
	y = df['price'].values.reshape(-1,1)

	# -- Normalization of the data prior to the fit process
	# -- Normalization not necessary in case of covariance/normal equation
	if method == "gradient-descent":
		print("Data standardization prior gradient descent run.")
		std_x = np.std(x)
		std_y = np.std(y)
		mean_x = np.mean(x)
		mean_y = np.mean(y)
		x = 0.5 * (x - np.mean(x)) / np.std(x)
		y = 0.5 * (y - np.mean(y)) / np.std(y)
	
	# -- Visualization
	if b_dynamic:
		cost_x = [0]
		cost_y = [cost(predict(x, theta), y)]
		lst_theta = np.array([])
		animate_visu()
	if b_static:
		theta = static_visu(x, y, theta, method)
	if b_console:
		theta = console_visu(x, y, theta, method)

	# -- Verbose summary = print of theta and cost plus writing the theta in the JSON file.
	if method == "gradient-descent":
		print(CYAN, "Final value of theta (standardized context):".ljust(50), END, theta.reshape(-1,))
		print(CYAN, "Final cost (standardized context):".ljust(50), END, cost(predict(x,theta),y))
		X = np.array([[mean_x], [2 * std_x + mean_x]])
		X = np.hstack((np.ones((2,1)), X))
		Y = np.array([[2 * std_y * theta[0] + mean_y],
					  [2 * std_y * (theta[0] + theta[1]) + mean_y]]).reshape(-1,1)
		nonstd_theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T,Y))
		print(CYAN, "Final value of theta (non standardized context):".ljust(50), END, nonstd_theta.reshape(-1,))
		print(CYAN, "Final cost (non standardized context):".ljust(50), END,
			cost(predict(df['km'].values.reshape(-1,1),nonstd_theta),
				df['price'].values.reshape(-1,1)))
		open_write_coeff(nonstd_theta)
	else:
		print(CYAN, "Value of theta (non standardized context):".ljust(45), END, theta.reshape(-1,))
		print(CYAN, "Cost (non standardized context):".ljust(45), END, cost(predict(x,theta),y))
		open_write_coeff(theta)
