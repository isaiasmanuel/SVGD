#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:45:34 2025

@author: isaias
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy as sp
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import torch
torch.set_default_dtype(torch.float64)

os.chdir('/Users/isaias/Desktop/SVGD')

WasSize=10

###################################################################




# with open("./Gauss_Tiempo_1_500.pkl", 'rb') as f:
#     Tiempo1 = pickle.load(f)

# with open("./Gauss_1_500.pkl", 'rb') as f:
#     Cadena1 = pickle.load(f)


# with open("./Gauss_Tiempo_500_500.pkl", 'rb') as f:
#     Tiempo0 = pickle.load(f)

# with open("./Gauss_500_500.pkl", 'rb') as f:
#     Cadena0 = pickle.load(f)
    
###################################################################
with open("./Banana_Tiempo_1_500.pkl", 'rb') as f:
    Tiempo1 = pickle.load(f)

with open("./Banana_1_500.pkl", 'rb') as f:
    Cadena1 = pickle.load(f)


with open("./Banana_Tiempo_500_500.pkl", 'rb') as f:
    Tiempo0 = pickle.load(f)

with open("./Banana_500_500.pkl", 'rb') as f:
    Cadena0 = pickle.load(f)
    
###################################################################


# ########### Caso Banana



# degree= 10
# d=2
# s1=100
# MU=torch.zeros((3,d))
# MU[1,1]=5
# MU[2,:2]=15

# W=torch.tensor((0.4,0.4,0.2))
# W=W/torch.sum(W)


# b1=0.03
# b2=0.05
# b3=0.03
# b=np.array((b1,b2,b3))

# def mvt(a,b,mu1=torch.zeros(d)):
#     p1=(1+ (a[0]-mu1[0])**2/(100*degree))**((-degree+1)/2) 
#     p2=1/torch.sqrt((degree+ (a[0]-mu1[0])**2/100)/(degree+1))
#     p3=(1+1/(degree+ (a[0]-mu1[0])**2/100)*(a[1]+b*(a[0]-mu1[0])**2+100*b-mu1[1])**2+ torch.sum((a[2:]-mu1[2:])**2) )**(-(degree+d)/2)
#     return p1*p2*p3



# def Mezcla(a):
#   resultado=0
#   for i in range(len(MU)):
#     resultado+=W[i]*mvt(a,b[i],mu1=MU[i,:])
#   return torch.log(resultado)

# def gradfun(a):
#   a=torch.from_numpy(a)
#   a=a.to(torch.float)
#   a=torch.reshape(a,(d,))
#   a.requires_grad=True
#   Q=Mezcla(a)
#   Q.backward()
#   return a.grad


# def Wasserstein(X,p=2) :
#     Grupos=np.random.choice(len(W), len(X),p=W)
#     MontecarloT=np.zeros(0)
#     try:
#         X1=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==0),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
#         X1[:,1]=X1[:,1]-b[0]*X1[:,0]**2-100*b[0]
#     except:
#         print("Grupo Vacio")
#     # plt.scatter(X1[:,0],X1[:,1])
#     # plt.show()
#     try:
#         X2=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==1),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
#         X2[:,1]=(X2[:,1]+MU[1,1].numpy())-b[1]*X2[:,0]**2-100*b[1]
#     except:
#         print("Grupo Vacio")

#     try:
#         X3=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==2),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
#         X3[:,0]=X3[:,0]+MU[2,0].numpy()
#         X3[:,1]=X3[:,1]+MU[2,1].numpy()-b[2]*(X3[:,0]-MU[2,0].numpy())**2-100*b[2]
#     except:
#         print("Grupo Vacio")

#     MontecarloT=np.vstack((X1,X2,X3))
    
#     Costos= cdist(MontecarloT, X, metric='minkowski', p=p) ** p
#     xAxis,yAxis=linear_sum_assignment(Costos)

#     return (np.sum(Costos[xAxis,yAxis])/len(X))**(1/p)



# def Grafica(Cadena,TiempoEjecucion):
#     np.random.seed(1)
#     SerieTiempo=np.ones((len(Cadena),WasSize))
    
#     for i in range(len(Cadena)):
#         for j in range(SerieTiempo.shape[1]):
#             SerieTiempo[i,j]=Wasserstein(Cadena[i],p=2)
            
#         print(i/len(Cadena))
    
#     return SerieTiempo



#########Caso Gauss

MU=2*torch.hstack((torch.zeros((5,1)),torch.reshape(torch.arange(5),(5,1))))
for i in range(4):
  MU=torch.vstack((MU,2*torch.hstack(((i+1)*torch.ones((5,1)),torch.reshape(torch.arange(5),(5,1))))))

W=np.arange(25)+1
W=W/np.sum(W)

S1I=torch.tensor(((5.,0.),(0.,5.)))
# det1=1/100

a = torch.tensor([[1., 1.]], requires_grad=True)

def Mezcla(a):
  resultado=0
  for i in range(len(MU)):
    resultado+=W[i]*mvn(a,MU[i,:],S1I)
  return torch.log(resultado)

#Q=Mezcla(a)
#Q.backward()
#print(a.grad)


def gradfun(a):
  a=torch.from_numpy(a)
  a=a.to(torch.float)
  a=torch.reshape(a,(1,2))
  a.requires_grad=True
  Q=Mezcla(a)
  Q.backward()
  return a.grad

def Wasserstein(X,p=2) :
    Grupos=np.random.choice(len(W), len(X),p=W)
    MontecarloT=np.zeros((len(Grupos),2))    
    for i in range(len(W)):
        MontecarloT[Grupos==i]=sp.stats.multivariate_normal.rvs(size=np.sum(Grupos==i),mean=MU[i],cov=np.linalg.inv(S1I) )
    
    Costos= cdist(MontecarloT, X, metric='minkowski', p=p) ** p
    xAxis,yAxis=linear_sum_assignment(Costos)


    return (np.sum(Costos[xAxis,yAxis])/len(X))**(1/p)


def Grafica(Cadena,TiempoEjecucion):
    np.random.seed(1)
    SerieTiempo=np.ones((len(Cadena),WasSize))
    
    for i in range(len(Cadena)):
        for j in range(SerieTiempo.shape[1]):
            SerieTiempo[i,j]=Wasserstein(Cadena[i],p=2)
            
        print(i/len(Cadena))
    
    return SerieTiempo
    



##########################################################################################


    



def GraficaSerieTiempo(SerieTiempo,TiempoEjecucion,tipo):
    if tipo==0:
        for i in range(WasSize):
            plt.plot(TiempoEjecucion[:-1],(SerieTiempo[:-1,i]), alpha=0.1, color="blue")
            plt.plot(TiempoEjecucion, np.apply_along_axis(np.median,1,(SerieTiempo)), color="black",alpha=0.5)
    elif tipo==1:
        for i in range(WasSize):
            plt.plot(TiempoEjecucion,(SerieTiempo[:,i]), alpha=0.1, color="orange")
            plt.plot(TiempoEjecucion, np.apply_along_axis(np.median,1,(SerieTiempo)), color="green", alpha=0.5)

    
    
    
SerieTiempo0=Grafica(Cadena0,Tiempo0)
SerieTiempo1=Grafica(Cadena1,Tiempo1)



import matplotlib
matplotlib.rc('font', size=16)

# GraficaSerieTiempo(SerieTiempo0,np.cumsum(Tiempo0),1)
# plt.show()

GraficaSerieTiempo(np.vstack((SerieTiempo0,SerieTiempo0[-1,:])),np.hstack((np.cumsum(Tiempo0),np.cumsum(Tiempo1)[-1])),0)
GraficaSerieTiempo(SerieTiempo1,np.cumsum(Tiempo1),1)
plt.axvline(np.cumsum(Tiempo0)[-1],alpha=0.5, linestyle="dashed")
plt.grid(True) 
plt.xlabel("Seconds")
plt.ylabel(r"$W_\pi$")
plt.show()


TamanioMuestra=np.zeros(len(Cadena1))

for i in range(len(Cadena1)):
    TamanioMuestra[i]=len(Cadena1[i])




plt.plot(np.cumsum(Tiempo1),TamanioMuestra)
plt.xlabel("Seconds")
plt.ylabel(r"$n$")
plt.grid(True) 
plt.show()


# plt.scatter(Cadena0[-1][:,0],Cadena0[-1][:,1])
# plt.show()
# plt.scatter(Cadena1[-1][:,0],Cadena1[-1][:,1])
# plt.show()


xmin= np.min((np.min(Cadena0[-1][:,0]),np.min(Cadena1[-1][:,0])))
xMax= np.max((np.max(Cadena0[-1][:,0]),np.max(Cadena1[-1][:,0])))
ymin= np.min((np.min(Cadena0[-1][:,1]),np.min(Cadena1[-1][:,1])))
yMax= np.max((np.max(Cadena0[-1][:,1]),np.max(Cadena1[-1][:,1])))


sns.set_style("white")
sns.kdeplot(x=Cadena0[-1][:,0], y=Cadena0[-1][:,1], fill=True,bw_adjust=0.25) #0.4 Gaussian case
# plt.xlim(xmin-5,xMax+5)
# plt.ylim(ymin-5,yMax+5)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(True) 
plt.show()




sns.set_style("white")
sns.kdeplot(x=Cadena1[-1][:,0], y=Cadena1[-1][:,1], fill=True,bw_adjust=0.25)
# plt.xlim(xmin-5,xMax+5)
# plt.ylim(ymin-5,yMax+5)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(True) 
plt.show()







