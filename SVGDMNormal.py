#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:00:39 2024

@author: isaias
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def mvn(a,mu1,S1I,det1):
    densidad=torch.exp(-(a-mu1)@S1I@torch.transpose(a-mu1,0,1)/2)/(2*np.pi*det1)
    return densidad[0][0]

mu1=np.array([[6.,0.]])
S1=np.array(((16.,16.),(16.,25.)))
mu2=np.array([[-3.,10.]])
S2=np.array(((1.,0.1),(0.1,1.)))
S1I=np.linalg.inv(S1)
S2I=np.linalg.inv(S2)
det1=np.linalg.det(S1)
det2=np.linalg.det(S2)
w=np.array((0.7,0.3))



import torch
a = torch.tensor([1., 1.], requires_grad=True)

mu1= torch.from_numpy(mu1)
S1I= torch.from_numpy(S1I)
Q=mvn(a,mu1,S1I,det1)
Q.backward()
a.grad
(a-mu1).shape





MU=torch.hstack((torch.zeros((5,1)),torch.reshape(torch.arange(5),(5,1))))
for i in range(4):
  MU=torch.vstack((MU,torch.hstack(((i+1)*torch.ones((5,1)),torch.reshape(torch.arange(5),(5,1))))))

W=np.arange(25)+1
W=W/np.sum(W)

S1I=torch.tensor(((10.,0.),(0.,10.)))
det1=1/100

a = torch.tensor([[1., 1.]], requires_grad=True)

def Mezcla(a):
  resultado=0
  for i in range(len(MU)):
    resultado+=W[i]*mvn(a,MU[i,:],S1I,det1)
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

#gradfun(X[s,:])[0]




def SVGD(iteraciones, X):
    historia=np.zeros(0) #Aqui se va a guardar una serie de tiempo de las normas que eventualmente puede ser usada como criterio de paro
    for m in range(iteraciones):
        Xaux=np.copy(X)
        #X=torch.from_numpy(X)
        
        n=len(X)
        KJI=np.zeros((n,n))    #Lo hice de forma matricial para evitar for
        # GradLogP=np.zeros((n,param))
        GradKJI=np.zeros((n,n,X.shape[1]))



        for j in range(len(X)):
            for i in np.arange(j,len(X),1):
                (Kval,grad)=K(X[j,:],X[i,:])
                KJI[j,i]=Kval
                GradKJI[j,i,:]=grad

        for i in range(len(X)):
            for j in np.arange(i,len(X),1):
                KJI[j,i]=KJI[i,j] #Aprovecho que es simetrica
                GradKJI[j,i,:]=-GradKJI[i,j,:] #Aprovecho que es antisimetrica



        #GradLogP=np.reshape(np.apply_along_axis(gradlog, 1, X),(n,2))
        GradLogP=np.zeros((1,2))
        for s in range(len(X)):
          GradLogP=np.vstack((GradLogP,gradfun(X[s,:])[0]))
        GradLogP=GradLogP[1:,:]

        SumaFinal=np.zeros((n,X.shape[1]))  #El segundo sumando del algoritmo (en phi)
        for i in range(n):
            SumaFinal[i,:]=np.apply_along_axis(np.mean, 0, GradKJI[:,i,:])

        X=X+epsi*(KJI.T@GradLogP/n+SumaFinal) #Actualizo


        historia=np.hstack((historia,np.linalg.norm(Xaux-X))) #Algunas graficas para ir viendo la evolucion
        #Si se comenta de historia a plt.show es mucho mas rapido pero no vemos los cambios
        if m%100==0:  #El 10 es para que cada ese numero de iteraciones nos muestre las graficas
        # plt.plot()
            fig, axs = plt.subplots(2,2)
            fig.suptitle(str(len(X))+", "+str(np.round(historia[-1],5)))
            axs[0,0].hist(X[:,0])
            axs[0,1].hist(X[:,1])
            axs[1,0].scatter(X[:,0],X[:,1],c=Estatus)
            axs[1,1].scatter(np.arange(m+1),-np.log(historia))
            fig.tight_layout()
            plt.show()
        if np.sum((Xaux-X)**2)/n<0.0001/n:
          break

    return X
    # if np.linalg.norm(Xaux-X)<epsilonTol :
    #     break

########Kernel y su gradiente
def K(X,Y):
    rk=np.exp(-(X-Y).T@(X-Y)/r)
    grk=-2*(X-Y)*rk/r
    return (rk,grk)



r=1  #Constante del reproducing kernel
epsi=10**(-1)



##############

X=np.random.normal(size=(2,2))

Estatus=np.ones(len(X))
Estatus[0]=0

plt.plot()
plt.scatter(X[:,0],X[:,1])
plt.show()

####################





for s in range(100):
    iteraciones=np.random.poisson(lam=len(X)*5,size=1)[0]
    start = time.time()

    X=SVGD(500, X)

    end = time.time()
    print(len(X), end - start)

    AgregarNuevos=np.ones((1,2))
    EstatusNuevos=np.ones((0))
    for i in np.arange(len(Estatus)):
        if Estatus[i]==0:
            nuevos=np.random.choice((1,2,3),p=(1/3,1/3,1/3))
            nuevos=np.random.normal(size=(nuevos,2))*1+X[i,:]
            Estatus[i]=2
            EstatusArgegar=np.ones(len(nuevos))
            EstatusArgegar[0]=0
            EstatusNuevos=np.hstack((EstatusNuevos,EstatusArgegar))
            AgregarNuevos=np.vstack((AgregarNuevos,nuevos))


        if Estatus[i]==1:
            nuevos=np.random.choice((0,1,2),p=(0.5,0.2,0.3))
            if nuevos==0:
                Estatus[i]=2
            else :
                nuevos=np.random.normal(size=(nuevos,2))*1+X[i,:]
                Estatus[i]=2
                EstatusArgegar=np.ones(len(nuevos))
                EstatusNuevos=np.hstack((EstatusNuevos,EstatusArgegar))
                AgregarNuevos=np.vstack((AgregarNuevos,nuevos))


    X=np.vstack((X,AgregarNuevos[1:,:]))
    Estatus=np.hstack((Estatus,EstatusNuevos))





x = np.linspace(-2,6,100)
y = np.linspace(-2,6,100)
Xg,Yg = np.meshgrid(x,y)

pdf = np.zeros(Xg.shape)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Obs=torch.tensor((Xg[i,j], Yg[i,j]))
        Obs=Obs.to(torch.float)
        Obs=Obs.reshape((1,2))
        pdf[i,j] = Mezcla(Obs)



# plt.contourf(Xg, Yg, np.log10(pdf), cmap='viridis')
# plt.contourf(Xg, Yg, pdf, cmap='viridis')
plt.plot()
plt.contour(Xg, Yg, np.exp(pdf), cmap='viridis',levels=1000)
plt.colorbar()
plt.show()



x2 = np.linspace(-1,5,20)
y2 = np.linspace(-1,5,20)
Xg2,Yg2 = np.meshgrid(x2,y2)


pdf0 = np.zeros(Xg2.shape)
pdf1 = np.copy(pdf0)
for i in range(Xg2.shape[0]):
    for j in range(Xg2.shape[1]):
        Obs=np.array((Xg2[i,j], Yg2[i,j]))
        pdf0[i,j] = gradfun(Obs)[0][0]
        pdf1[i,j] = gradfun(Obs)[0][1]

plt.quiver(Xg2,Yg2,pdf0,pdf1)

import seaborn as sns
sns.set_style("white")
sns.kdeplot(x=X[:,0], y=X[:,1], fill=True,bw_adjust=0.4)
plt.show()



















