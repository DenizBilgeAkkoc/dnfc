# GRASP POINT YAXIS JACOBIAN
# Try: [J,~]=Jyaxis_left_GRASP.compute_jac(0,0, 0,0,0,0,0,0,0, 0)
import numpy as np

class Jyaxis_left_GRASP:
	@staticmethod
	def compute_jac(t1, t2, a1, a2, a3, a4, a5, a6, a7, g):
		J = np.zeros([3,10]);
		J[0,0] = 1.0*np.sin(a7)*(np.sin(a6)*(1.0*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))) + np.cos(a6)*(1.0*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))))) - 1.0*np.cos(a7)*(np.sin(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.cos(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))))
		J[0,1] = - 1.0*np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)))) - 1.0*np.sin(a7)*(np.sin(a6)*(np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)))) + np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) + np.cos(a4)*np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))))
		J[0,2] = np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)))) + np.sin(a7)*(np.sin(a6)*(np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)))) + np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))) + np.cos(a4)*np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2))))
		J[0,3] = - 1.0*np.cos(a7)*(np.cos(a5)*(np.sin(a4)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - 1.0*np.cos(a3)*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.sin(a3)*np.sin(a5)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) - np.sin(a7)*(1.0*np.sin(a6)*(np.sin(a5)*(np.sin(a4)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - 1.0*np.cos(a3)*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) - 1.0*np.cos(a5)*np.sin(a3)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) - np.cos(a6)*(np.cos(a4)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a3)*np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))
		J[0,4] = np.sin(a7)*(np.sin(a6)*(np.cos(a5)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.cos(a4)*np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) + np.cos(a6)*np.sin(a4)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a7)*(np.sin(a5)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) - 1.0*np.cos(a4)*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))
		J[0,5] = np.sin(a7)*(np.cos(a6)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) + 1.0*np.sin(a5)*np.sin(a6)*(1.0*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))) + 1.0*np.cos(a5)*np.cos(a7)*(1.0*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))
		J[0,6] = np.sin(a6)*np.sin(a7)*(np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.cos(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))) - 1.0*np.cos(a7)*(np.sin(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))
		J[0,7] = 1.0*np.sin(a7)*(np.sin(a6)*(1.0*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) + 1.0*np.cos(a6)*(np.sin(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))))
		J[0,8] = - 1.0*np.sin(a7)*(np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.cos(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))) - np.cos(a7)*(np.cos(a6)*(1.0*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - np.sin(a6)*(np.sin(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))))
		J[0,9] = 0.0
		J[1,0] = np.cos(a7)*(np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.cos(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))) - np.sin(a7)*(np.cos(a6)*(1.0*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - np.sin(a6)*(np.sin(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))))
		J[1,1] = - 1.0*np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) - 1.0*np.sin(a7)*(np.sin(a6)*(np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) + np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))))
		J[1,2] = np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) + np.sin(a7)*(np.sin(a6)*(np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) + np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))))
		J[1,3] = np.cos(a7)*(np.cos(a5)*(np.sin(a4)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))) - 1.0*np.sin(a3)*np.sin(a5)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))) - 1.0*np.sin(a7)*(np.cos(a6)*(np.cos(a4)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - 1.0*np.cos(a3)*np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))) - 1.0*np.sin(a6)*(np.sin(a5)*(np.sin(a4)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))) + np.cos(a5)*np.sin(a3)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))))
		J[1,4] = 1.0*np.cos(a7)*(np.sin(a5)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + 1.0*np.cos(a4)*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) - np.sin(a7)*(np.sin(a6)*(np.cos(a5)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.cos(a4)*np.sin(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) - np.cos(a6)*np.sin(a4)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))))
		J[1,5] = 1.0*np.cos(a5)*np.cos(a7)*(1.0*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))) - np.sin(a7)*(np.cos(a6)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))) - 1.0*np.sin(a5)*np.sin(a6)*(1.0*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))))
		J[1,6] = 1.0*np.cos(a7)*(1.0*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))) + np.sin(a6)*np.sin(a7)*(np.sin(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.cos(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1))))
		J[1,7] = -np.sin(a7)*(np.cos(a6)*(1.0*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))) - 1.0*np.sin(a6)*(1.0*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))))
		J[1,8] = - 1.0*np.sin(a7)*(np.sin(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.cos(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))) - np.cos(a7)*(np.sin(a6)*(1.0*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) + np.sin(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))) + np.cos(a6)*(1.0*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))))
		J[1,9] = 0.0
		J[2,0] = 0.0
		J[2,1] = - 1.0*np.sin(a7)*(np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) - 1.0*np.cos(a4)*np.sin(a2)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + 1.0*np.sin(a6)*(1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) - np.cos(a5)*(np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))))) - 1.0*np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a5)*(np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))))
		J[2,2] = np.sin(a7)*(np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) - 1.0*np.cos(a4)*np.sin(a2)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + 1.0*np.sin(a6)*(1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) - np.cos(a5)*(np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))))) + np.cos(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a2)*np.sin(a4)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))) + np.sin(a5)*(np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a2)*np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2))))
		J[2,3] = np.cos(a7)*(np.cos(a5)*(np.cos(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a3)*np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a3)*np.sin(a5)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.sin(a7)*(np.sin(a6)*(np.sin(a5)*(np.cos(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + np.cos(a3)*np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a5)*np.sin(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.cos(a6)*(np.cos(a2)*np.cos(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 1.0*np.cos(a3)*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))))
		J[2,4] = np.cos(a7)*(np.sin(a5)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.cos(a4)*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))) - 1.0*np.sin(a7)*(np.sin(a6)*(np.cos(a5)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))) + np.cos(a6)*np.sin(a4)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))))
		J[2,5] = np.cos(a5)*np.cos(a7)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a7)*(np.cos(a6)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*np.sin(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))))
		J[2,6] = - 1.0*np.cos(a7)*(np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))) - 1.0*np.sin(a6)*np.sin(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))))
		J[2,7] = np.sin(a7)*(np.sin(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a6)*(np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))))
		J[2,8] = np.sin(a7)*(np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))) - 1.0*np.cos(a7)*(np.cos(a6)*(np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a6)*(np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))))
		J[2,9] = 0.0
		return J
	#end of compute Jacobian
#end of class Jyaxis_left_GRASP
def main():
	print(Jyaxis_left_GRASP.compute_jac(0,0, 0,0,0,0, 0,0,0, 0))

if __name__ == '__main__':
	main()

