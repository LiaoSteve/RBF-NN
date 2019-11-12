import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import timeit
import time
import datetime
from tqdm import tqdm

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Gaussian(input_array, m, sigma):    
    return np.exp( - np.linalg.norm(input_array-m, axis=1, keepdims=True)**2 / (2 * sigma**2) ) 

np.random.seed(6666)
x1 = np.linspace(-5,5, 400)
x2 = np.linspace(-5,5, 400)
np.random.shuffle(x1) 
np.random.shuffle(x2) 

d = x1**2 + x2**2    

# Normalize d to range 0.2~0.8
d_max = np.max(d)
d_min = np.min(d)
d = (d-d_min)/(d_max-d_min)*(0.8-0.2)+0.2 

#---------------- Input vector -----------------------------
num_in  = 2


#---------------- Radial Basis phi -----------------------
num_phi = 3
s_o     = sigma = np.random.uniform(0,2,[num_phi,1])
phi_out = np.zeros([num_phi,1])
m_o     = m = np.random.uniform(-1,1,[num_phi,num_in])
#---------------- Output ---------------------------------
num_out  = 1
bias_out = np.random.uniform(-0.5,0.5,[num_out,1])
w_out    = np.random.uniform(-0.5,0.5,[num_phi,num_out])

#---------------- Parameter --------------------------
eta   = 1
mom   = 0.95
epoch = 1000

Eav_train = np.zeros([epoch])
Eav_test = np.zeros([epoch])

dw_out    = temp1 = np.zeros([num_phi,num_out]) 
dbias_out = temp2 = np.zeros([num_out,1])

dm        = temp3 = np.zeros([num_phi,num_in])
dsigma    = temp4 = np.zeros([num_phi,1])

#---------------- Traning ----------------------------
t0 = timeit.default_timer()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pbar = tqdm(total =epoch)
plt.ion()
plt.show()

for i in range(epoch):         

    #--------------- Feed Forward  -------------------    
    e = np.zeros([300])
    E_train = np.zeros([300])
    for j in range(300):
        X   = np.array([x1[j],x2[j]]).reshape(2,1)

        XX = np.array(X)
        for _ in range(num_phi-1):
            XX = np.append(XX, X, axis=1)
        XX = np.transpose(XX)
        dx_m = XX - m
        
        phi_out = Gaussian(XX,m,sigma)
        out = sigmoid(np.dot(np.transpose(phi_out),w_out) + bias_out)

        #--------------- Back Propagation-----------------
        e[j]   = (d[j]-out) 
        E_train[j] = 0.5 * e[j]**2
        locg_k = e[j] * (out*(1-out))
        temp2  = temp2 + mom * dbias_out + eta * locg_k * 1     
        temp1  = temp1 + mom * dw_out + eta * locg_k * phi_out                  

        temp3  = temp3 + mom * dm + eta * e[j] * w_out * phi_out / sigma**2 * (dx_m)              
        temp4  = temp4 + mom * dsigma + eta * e[j] * w_out * phi_out / sigma**3 * np.linalg.norm(dx_m, axis=1, keepdims=True)

    #---------- Average delta weight -----------------
    dbias_out = temp2/300
    dw_out    = temp1/300       
    dm        = temp3/300
    dsigma    = temp4/300  

    temp1 = np.zeros([num_phi,num_out]) 
    temp2 = np.zeros([num_out,1])
    temp3 = np.zeros([num_phi,num_in])
    temp4 = np.zeros([num_phi,1])

    #----------  New weight --------------------------
    bias_out  = bias_out + dbias_out
    w_out     = w_out + dw_out              
    m         = m + dm
    sigma     = sigma + dsigma

    #---------- Eave_train
    Eav_train[i]  = np.mean(E_train)

    #---------- Test data loss ---------------     
    E_test = np.zeros([100])
    for j in range(100):
        X   = np.array([x1[300+j],x2[300+j]]).reshape(2,1)

        XX = np.array(X)
        for _ in range(num_phi-1):
            XX = np.append(XX, X, axis=1)
        XX = np.transpose(XX)
        dx_m = XX - m

        phi_out = Gaussian(XX,m,sigma)
        out = sigmoid(np.dot(np.transpose(phi_out),w_out) + bias_out)
        E_test = 0.5*( d[300+j] - out )**2      

    Eav_test[i] = np.mean(E_test)
    if i % 1000 == 0 and i!=0:
        pbar.update(1000)
    if i% 50==0:
        y_predict = np.zeros([100])    
        for k in range(100):
            X      = np.array([x1[300+k],x2[300+k]]).reshape(2,1)
            XX = np.array(X)
            for _ in range(num_phi-1):
                XX = np.append(XX, X, axis=1)
            XX = np.transpose(XX)     

            phi_out = Gaussian(XX,m,sigma)
            out = sigmoid(np.dot(np.transpose(phi_out),w_out) + bias_out)
            y_predict[k] = out

        y_predict = (y_predict-0.2)/(0.8-0.2)*(d_max-d_min)+d_min
        fig = plt.figure(num='Animation')
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x1[300:], x2[300:], y_predict[:],c='g', marker='o', s=15).findobj()        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('y')
        plt.title('Predict Data : y = x1^2 +x2^2')        
        plt.pause(0.001)

pbar.close()
t1 = (timeit.default_timer()-t0)
print('Training time: {} min'.format((t1/60)))

#--------- Predict data --------------
y_predict = np.zeros([100])
E_predict = np.zeros([100])
for j in range(100):
        X      = np.array([x1[300+j],x2[300+j]]).reshape(2,1)

        XX = np.array(X)
        for _ in range(num_phi-1):
            XX = np.append(XX, X, axis=1)
        XX = np.transpose(XX)
        dx_m = XX - m

        phi_out = Gaussian(XX,m,sigma)
        out = sigmoid(np.dot(np.transpose(phi_out),w_out) + bias_out)
        y_predict[j] = out
        E_predict[j] = 0.5*( d[300+j] - out )**2
Eav_predict = np.mean(E_predict)

#----------- Return the data they were normolized before ----------------------
y_predict = (y_predict-0.2)/(0.8-0.2)*(d_max-d_min)+d_min  


#------------ Record the result ------------------

import csv
table = [
    #['TimeStamp','Unit', 'Eta', 'Alpha','Training_loss','Predict_loss','Epoch','Time(min)'],
    [ now,num_phi, eta, mom, Eav_train[epoch-1], Eav_predict,epoch, int(t1/60)]    
]
with open('RBF_output.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)

#------------ Scattering Data set ----------------------
#------- Return the data they were normolized before ---
d = (d-0.2)/(0.8-0.2)*(d_max-d_min)+d_min 

fig1 = plt.figure(num ='Data_Set', figsize=(10,5))
ax = fig1.add_subplot(121, projection='3d')
ax.scatter(x1[:300], x2[:300], d[:300], c='b', marker='o', s=5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('Training Data')

ax2 = fig1.add_subplot(122, projection='3d')
ax2.scatter(x1[300:], x2[300:], d[300:], c='r', marker='o', s=5)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
plt.title('Testing Data')
plt.show()



#------------ Scattering y_Predict data -----------------
fig2 = plt.figure(num = now, figsize=(14,6))
ax1 = fig2.add_subplot(122, projection='3d')
ax1.scatter(x1[300:], x2[300:], y_predict[:],c='g', marker='o', s=15)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
plt.title('Predict Data : y = x1^2 +x2^2')

#------------ plot training and testing Loss -----------------
ax3 = fig2.add_subplot(121)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')
ax3.plot(range(epoch),Eav_train,label='Train Set :'+str(Eav_train[epoch-1]))
ax3.plot(range(epoch),Eav_test, color='red', linewidth=1.0, linestyle='--', label='Test Set :'+ str(Eav_test[epoch-1]))
plt.legend(loc='upper right')
plt.title('Unit:'+str(num_phi)+', Eta:'+str(eta)+', Alpha:'+str(mom))
plt.show()


#----------- Plot init Gaussian -------------------------
if num_phi ==3:
    x1 = np.linspace(-5,5,400)
    x2 = np.linspace(-5,5,400)
    G = np.zeros([400,400,num_phi])
    [X1,X2] = np.meshgrid(x1,x2)

    for i in range(400) :
        for j in range(400):
            X   = np.array([X1[i,j],X2[i,j]]).reshape(2,1)
            XX = np.array(X)
            for _ in range(num_phi-1):
                XX = np.append(XX, X, axis=1)
            XX = np.transpose(XX)
            phi_out = Gaussian(XX,m_o,s_o)
            G[i,j,:] = phi_out[:,0]         


    fig3 = plt.figure(num ='Gaussian_init',figsize=(17,6))
    plt.suptitle('Before training ...',fontsize = 16)

    ax = fig3.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, G[:,:,0],rstride=10, cstride=10, cmap='rainbow')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    plt.title('Gaussian 1')

    ax2 = fig3.add_subplot(132, projection='3d')
    ax2.plot_surface(X1, X2, G[:,:,1],rstride=10, cstride=10, cmap='rainbow')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    plt.title('Gaussian 2')

    ax3 = fig3.add_subplot(133, projection='3d')
    ax3.plot_surface(X1, X2, G[:,:,2],rstride=10, cstride=10, cmap='rainbow')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')

    plt.title('Gaussian 3')

    plt.show()

    #-------------- Plot trained Gaussian ----------- 

    for i in range(400) :
        for j in range(400):
            X   = np.array([X1[i,j],X2[i,j]]).reshape(2,1)
            XX = np.array(X)
            for _ in range(num_phi-1):
                XX = np.append(XX, X, axis=1)
            XX = np.transpose(XX)
            phi_out = Gaussian(XX,m,sigma)
            G[i,j,:] = phi_out[:,0]         


    fig3 = plt.figure(num ='Gaussian_trained',figsize=(17,6))
    plt.suptitle('After training ...',fontsize = 16)

    ax = fig3.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, G[:,:,0],rstride=10, cstride=10, cmap='rainbow')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.title('Gaussian 1')

    ax2 = fig3.add_subplot(132, projection='3d')
    ax2.plot_surface(X1, X2, G[:,:,1],rstride=10, cstride=10, cmap='rainbow')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    plt.title('Gaussian 2')

    ax3 = fig3.add_subplot(133, projection='3d')
    ax3.plot_surface(X1, X2, G[:,:,2],rstride=10, cstride=10, cmap='rainbow')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    plt.title('Gaussian 3')
    plt.show()

plt.ioff()
plt.show()