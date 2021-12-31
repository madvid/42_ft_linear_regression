import numpy as np
from utils import simple_predict, open_read_coeff
import sys

RED = '\033[1;31m'
GREEN = '\033[1;32m'
END = '\033[0m'

# ########################################################################### #
# ___________________________________ MAIN __________________________________ #
# ########################################################################### #
if __name__ == "__main__":
	# -- Reading the coefficients theta in coefficients.json
	theta = open_read_coeff()
	
	# -- Reading of the input on the standard entry
	x = None
	while not isinstance(x, float):
		print("Give a distance (km):")
		x = sys.stdin.readline()
		if x == "":
			print(GREEN + "Ok, bye bye." + END)
			sys.exit()
		x = x[:-1]
		try:
			x = float(x)
			if x < 0.0:
				print("\tThe distance must be positive.")
				x = None
		except:
			print(RED + "\tx is not a float input." + END)
	
	# -- Calculation of the prediction based on theta coefficients and
	# -- the linear hypothesis
	pred = simple_predict(x,theta)
	print("Predicted price is:".ljust(25) + f"{pred[0]}")