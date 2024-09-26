# FORWARD KINEMATICS 
# Try: T=FK_left_WRIST.forward_ik(0,0, 0,0,0,0,0,0,0, 0) # rotm2quat(T[1:3,1:3])
import numpy as np

class FK_left_WRIST:
	@staticmethod
	def forward_ik(t1, t2, a1, a2, a3, a4, a5, a6, a7, g):
		T=np.zeros([4,4]);
		T[0,0] = np.sin(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))) - 1.0*np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))
		T[0,1] = np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) - 1.0*np.sin(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) + np.cos(a5)*(np.sin(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))))
		T[0,2] = np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) - 1.0*np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))))
		T[0,3] = 0.285*np.cos(t1)*np.sin(t2) - 0.25*np.cos(a2)*np.sin(t1) - 0.195*np.sin(t1) - 0.25*np.cos(a4)*(np.cos(a2)*np.sin(t1) + np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2))) + 0.25*np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.cos(t1)*np.cos(t2) + np.sin(a1)*np.cos(t1)*np.sin(t2)) + np.cos(a3)*(np.sin(a2)*np.sin(t1) - 1.0*np.cos(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)))) - 0.25*np.sin(a2)*(np.cos(a1)*np.cos(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t1)*np.cos(t2)) - 0.02*np.cos(a1)*np.cos(t1)*np.sin(t2) + 0.02*np.sin(a1)*np.cos(t1)*np.cos(t2)
		T[1,0] = - 1.0*np.cos(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))
		T[1,1] = np.sin(a5)*(np.sin(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) + np.cos(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.cos(a5)*(np.cos(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)))
		T[1,2] = np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) + 1.0*np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1)))
		T[1,3] = 0.195*np.cos(t1) + 0.285*np.sin(t1)*np.sin(t2) - 0.25*np.cos(a4)*(np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) - 1.0*np.cos(a2)*np.cos(t1)) - 0.25*np.sin(a4)*(1.0*np.cos(a3)*(np.sin(a2)*np.cos(t1) + np.cos(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1))) - np.sin(a3)*(np.sin(a1)*np.sin(t1)*np.sin(t2) + np.cos(a1)*np.cos(t2)*np.sin(t1))) - 0.25*np.sin(a2)*(np.cos(a1)*np.sin(t1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)*np.sin(t1)) + 0.25*np.cos(a2)*np.cos(t1) - 0.02*np.cos(a1)*np.sin(t1)*np.sin(t2) + 0.02*np.sin(a1)*np.cos(t2)*np.sin(t1)
		T[2,0] = np.cos(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))
		T[2,1] = - 1.0*np.cos(a5)*(np.cos(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a2)*np.sin(a4)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 1.0*np.sin(a5)*(np.cos(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) - 1.0*np.cos(a2)*np.sin(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)))
		T[2,2] = np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) + np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))
		T[2,3] = 0.285*np.cos(t2) - 0.02*np.sin(a1)*np.sin(t2) - 0.25*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) - 0.25*np.sin(a4)*(np.sin(a3)*(np.cos(a1)*np.sin(t2) - 1.0*np.sin(a1)*np.cos(t2)) + np.cos(a2)*np.cos(a3)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2))) - 0.02*np.cos(a1)*np.cos(t2) - 0.25*np.cos(a4)*np.sin(a2)*(np.sin(a1)*np.sin(t2) + np.cos(a1)*np.cos(t2)) + 1.03
		T[3,0] = 0.0
		T[3,1] = 0.0
		T[3,2] = 0.0
		T[3,3] = 1.0
		return T
	#end of computeForKin
#end of class FK_left_WRIST
def main():
	print(FK_left_WRIST.forward_ik(0,0, 0,0,0,0, 0,0,0, 0))

if __name__ == '__main__':
	main()

