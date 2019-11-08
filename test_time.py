import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import time
import datetime
from tqdm import tqdm

def Gassian(input_array, m, sigma):    
    return np.exp( -np.linalg.norm(input_array-m)**2 / (2*(sigma**2)) )    
t0 = timeit.default_timer()
X = np.linspace(-50,50,400)

D = np.exp(-(X-5)**2/(2*(5**2))) 
print(timeit.default_timer()-t0)


D = np.zeros([400])
t1 =timeit.default_timer()
for i in range (400):
    D[i] = Gassian(X[i],0,0.5)
print(timeit.default_timer()-t1)
plt.plot(X,D)
plt.show()