#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:56:39 2025

@author: isaias
"""

import torch

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
from multiprocessing import Pool, cpu_count, get_context, Process
import pickle

###############################
np.random.seed(1)

################################


b1=0.03
b2=0.05
b3=0.03
b=np.array((b1,b2,b3))


degree= 10
d=2
s1=100

sample=10**4


        
W=torch.tensor((0.4,0.4,0.2))
W=W/torch.sum(W)




################################

MU=torch.zeros((3,d))
MU[1,1]=5
MU[2,:2]=15

################################
x = np.linspace(-30,60,200) #Graficacion densidad
y = np.linspace(-30,45,200)
Xg,Yg = np.meshgrid(x,y)
pdf = np.zeros(Xg.shape)


################################
x2 = np.linspace(-30,60,20) #Graficacion gradiente
y2 = np.linspace(-30,45,20)
Xg2,Yg2 = np.meshgrid(x2,y2)

################################

sample=10**4
################################


r=1  #Constante del reproducing kernel
epsi=10**(-0) #Actualizacion del paso
################################
sd=5 #Desviacion del brinco
sdExplora=20
distmax=10
iteraciones=1000

sampleinicial=500
samplesize=500
epsmax=10**1
epsmin=10**(-1)

################################

def mvt(a,b,mu1=torch.zeros(d)):
    p1=(1+ (a[0]-mu1[0])**2/(100*degree))**((-degree+1)/2) 
    p2=1/torch.sqrt((degree+ (a[0]-mu1[0])**2/100)/(degree+1))
    p3=(1+1/(degree+ (a[0]-mu1[0])**2/100)*(a[1]+b*(a[0]-mu1[0])**2+100*b-mu1[1])**2+ torch.sum((a[2:]-mu1[2:])**2) )**(-(degree+d)/2)
    return p1*p2*p3



def Mezcla(a):
  resultado=0
  for i in range(len(MU)):
    resultado+=W[i]*mvt(a,b[i],mu1=MU[i,:])
  return torch.log(resultado)

def gradfun(a):
  a=torch.from_numpy(a)
  a=a.to(torch.float)
  a=torch.reshape(a,(d,))
  a.requires_grad=True
  Q=Mezcla(a)
  Q.backward()
  return a.grad


################################

Grupos=np.random.choice(np.arange(3), sample,p=(0.4,0.4,0.2))



X1=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==0),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
X1[:,1]=X1[:,1]-b[0]*X1[:,0]**2-100*b[0]
# plt.scatter(X1[:,0],X1[:,1])
# plt.show()


X2=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==1),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
X2[:,1]=(X2[:,1]+MU[1,1].numpy())-b[1]*X2[:,0]**2-100*b[1]

X3=sp.stats.multivariate_t.rvs(size=np.sum(Grupos==2),loc=np.zeros(d),shape=np.diag(np.append(100,np.ones(d-1))),df=degree)
X3[:,0]=X3[:,0]+MU[2,0].numpy()
X3[:,1]=X3[:,1]+MU[2,1].numpy()-b[2]*(X3[:,0]-MU[2,0].numpy())**2-100*b[2]
MontecarloT=np.vstack((X1,X2,X3))

# plt.scatter(MontecarloT[:,0],MontecarloT[:,1])
# plt.show()

################################ Graficas d=2


# ################################
x = np.linspace(-40,60,200) #Graficacion densidad
y = np.linspace(-40,25,200)
Xg,Yg = np.meshgrid(x,y)
pdf = np.zeros(Xg.shape)


################################
x2 = np.linspace(-40,60,30) #Graficacion gradiente
y2 = np.linspace(-40,25,30)
Xg2,Yg2 = np.meshgrid(x2,y2)

################################

if d==2:
    pdf = np.zeros(Xg.shape)
    
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            a=torch.tensor((Xg[i,j],Yg[i,j]))
            pdf[i,j] = Mezcla(a)
    
    plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis')
    # plt.scatter(MontecarloT[:,0],MontecarloT[:,1],s=0.1)
    plt.xlim(-40,20)
    plt.ylim(-40,45)
    plt.show()
    
    
    pdf0 = np.zeros(Xg2.shape)
    pdf1 = np.copy(pdf0)
    for i in range(Xg2.shape[0]):
        for j in range(Xg2.shape[1]):
            Obs=np.array((Xg2[i,j], Yg2[i,j]))
            pdf0[i,j] = gradfun(Obs)[0]
            pdf1[i,j] = gradfun(Obs)[0]
    
    plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis',levels=300)
    plt.quiver(Xg2,Yg2,pdf0,pdf1,color="cyan")
    plt.show()

################################

TiempoEjecucion=0
Cadena=[np.random.normal(size=(sampleinicial,2))]


def SVGD(iteraciones, X):
    historia=np.zeros(0) #Aqui se va a guardar una serie de tiempo de las normas que eventualmente puede ser usada como criterio de paro
    
    for m in range(iteraciones):
        inicio=time.time()
        # epsi=10-(10-10**(-0))*(m/iteraciones)
        epsi=epsmax-(epsmax-epsmin)*1/(1+np.exp(-0.01*(m-iteraciones*(1/2))))
        Xaux=np.copy(X)
        #X=torch.from_numpy(X)
        n=len(X)
        KJI=np.zeros((n,n))    #Lo hice de forma matricial para evitar for
        # GradLogP=np.zeros((n,param))
        GradKJI=np.zeros((n,n,X.shape[1]))



        for j in range(len(X)):
            for i in np.arange(j,len(X),1):
                #if np.linalg.norm(X[j,:]-X[i,:])<distmax:
                (Kval,grad)=K(X[j,:],X[i,:])
                KJI[j,i]=Kval
                GradKJI[j,i,:]=grad

        for i in range(len(X)):
            for j in np.arange(i,len(X),1):
                #if np.linalg.norm(X[j,:]-X[i,:])<distmax:
                KJI[j,i]=KJI[i,j] #Aprovecho que es simetrica
                GradKJI[j,i,:]=-GradKJI[i,j,:] #Aprovecho que es antisimetrica



        #GradLogP=np.reshape(np.apply_along_axis(gradlog, 1, X),(n,2))
        GradLogP=np.zeros((1,d))
        for s in range(len(X)):
          GradLogP=np.vstack((GradLogP,gradfun(X[s,:])))
        GradLogP=GradLogP[1:,:]

        SumaFinal=np.zeros((n,X.shape[1]))  #El segundo sumando del algoritmo (en phi)
        for i in range(n):
            SumaFinal[i,:]=np.apply_along_axis(np.mean, 0, GradKJI[:,i,:])

        X=X+epsi*(KJI.T@GradLogP/n+SumaFinal) #Actualizo


        historia=np.hstack((historia,np.linalg.norm(Xaux-X))) #Algunas graficas para ir viendo la evolucion
        
        final=time.time()
        
        #Si se comenta de historia a plt.show es mucho mas rapido pero no vemos los cambios
        if m%20==0:  #El 10 es para que cada ese numero de iteraciones nos muestre las graficas
            # plt.plot()
            fig, axs = plt.subplots(2,2)
            fig.suptitle(str(len(X))+", "+str(np.round(historia[-1],5)))
            axs[0,0].hist(X[:,0])
            axs[0,1].hist(X[:,1])
            if d==2:
                axs[1,0].contourf(Xg, Yg, np.exp(pdf), cmap='viridis')
            axs[1,0].scatter(X[:,0],X[:,1],c=Estatus,cmap='jet')
            # axs[1,0].set_xlim(-30,60)
            # axs[1,0].set_ylim(-30,45)
            axs[1,1].scatter(np.arange(m+1),-np.log(historia))
            fig.tight_layout()
            plt.show()
            # plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis')
            # plt.scatter(X[:,0],X[:,1],c=Estatus,cmap='jet')
            # plt.xlim(-30,60)
            # plt.ylim(-30,45)
            # plt.show()
        if np.sum((Xaux-X)**2)/n<0.001/n:
          break
      
        global TiempoEjecucion
        TiempoEjecucion=np.append(TiempoEjecucion,final-inicio)
        Cadena.append(X)
         

    return X
    # if np.linalg.norm(Xaux-X)<epsilonTol :
    #     break

########Kernel y su gradiente
def K(X,Y):
    rk=np.exp(-(X-Y).T@(X-Y)/r)
    grk=-2*(X-Y)*rk/r
    return (rk,grk)




##############

X=Cadena[0]

Estatus=np.ones(len(X))
Estatus[0]=0

# plt.plot()
# plt.scatter(X[:,0],X[:,1])
# plt.show()

####################



for s in range(100):
    if len(X)<samplesize:
        #np.random.poisson(lam=len(X)*5,size=1)[0]
        start = time.time()
    
        X=SVGD(iteraciones, X)
    
        end = time.time()
        print(len(X), end - start)
    
        AgregarNuevos=np.ones((1,d))
        EstatusNuevos=np.ones((0))
        for i in np.arange(len(Estatus)):
            if Estatus[i]==0:
                
                nuevos=np.random.choice((1,2,3),p=(1/3,1/3,1/3))
                nuevos=(np.random.normal(size=(nuevos,d))*sd+X[i,:])
                Estatus[i]=2
                EstatusArgegar=np.ones(len(nuevos))
                # EstatusArgegar[0]=0 Forzar a que un hijo de la rama sea la rama
                ############
                ############
                ############
                #Cuidado aqui
                ############
                ############
                EstatusNuevos=np.hstack((EstatusNuevos,EstatusArgegar))
                AgregarNuevos=np.vstack((AgregarNuevos,nuevos))
    
    
            if Estatus[i]==1:
                nuevos=np.random.choice((0,1,2),p=(0.5,0.2,0.3))
                if nuevos==0:
                    Estatus[i]=2
                else :
                    # explore=np.random.choice([0,1],size=1,p=(0.3,0.7))
                    explore=1
                    nuevos=(np.random.normal(size=(nuevos,d))*sd+X[i,:])*explore+(1-explore)*(np.random.normal(size=(nuevos,d))*sdExplora)
                    Estatus[i]=2
                    EstatusArgegar=np.ones(len(nuevos))
                    EstatusNuevos=np.hstack((EstatusNuevos,EstatusArgegar))
                    AgregarNuevos=np.vstack((AgregarNuevos,nuevos))
        if len(np.vstack((X,AgregarNuevos[1:,:])))>samplesize:
            break 
    
    
        X=np.vstack((X,AgregarNuevos[1:,:]))
        Estatus=np.hstack((Estatus,EstatusNuevos))
        Estatus[np.random.choice(len(Estatus))]=0
        
    else:
        X=SVGD(iteraciones, X)
        break 


with open("Banana_"+str(sampleinicial)+"_"+str(samplesize)+".pkl", 'wb') as f:
    pickle.dump(Cadena, f)    
        
with open("Banana_Tiempo_"+str(sampleinicial)+"_"+str(samplesize)+".pkl", 'wb') as f:
    pickle.dump(TiempoEjecucion, f)    
        




###################

# import seaborn as sns
# sns.set_style("white")
# sns.kdeplot(x=X[:,0], y=X[:,1], fill=True,bw_adjust=0.1)
# plt.xlim(-30,60)
# plt.ylim(-30,45)
# plt.show()

# sns.kdeplot(x=MontecarloT[:,0], y=MontecarloT[:,1], fill=True,bw_adjust=0.2)
# plt.xlim(-30,60)
# plt.ylim(-30,45)
# plt.show()

# import plotly.io as pio
# import plotly.express as px
# pio.renderers.default='browser'
# import plotly.express as px
# fig = px.scatter_matrix(X)
# fig.show()







