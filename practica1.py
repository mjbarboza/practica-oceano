# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
@author: Jimena
"""
import numpy as np
from Cargar import cargar
from matplotlib import pyplot as plt

K1 = 1.16e-5
K2 = 3.16e-5
K3 = 5.16e-5
beta= 10e-11
L=4000000
eps1= K1/(2*beta*L)#
eps2= K2/(2*beta*L)#es un orden menos
eps3= K3/(2*beta*L)# 
#ejercicio1

dir_salida='C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp1/'
dir_salida2='C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp2/'
dir_salida3='C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp3/'
Lx=4000000
Ly=2000000
nx=200
ny=100
beta= 10**-11

from Cargar import cargar
psi_temp1,vort_temp1,psiF1,vortF1,QG_diag1,QG_curlw1,X1,Y1,dx1,dy1=cargar(dir_salida,Lx,Ly,nx,ny)
psi_temp2,vort_temp2,psiF2,vortF2,QG_diag2,QG_curlw2,X2,Y2,dx2,dy2=cargar(dir_salida2,Lx,Ly,nx,ny)
psi_temp3,vort_temp3,psiF3,vortF3,QG_diag3,QG_curlw3,X3,Y3,dx3,dy3=cargar(dir_salida3,Lx,Ly,nx,ny)

Tau= 0.25
D=2000 
U= (2*np.pi*Tau)/(1025*(beta)*(Lx))
corriente1= psiF1*U*Lx#dimensionalizo funcion corriente
corriente2= psiF2*U*Lx
corriente3= psiF3*U*Lx
vort1=vortF1*(U/Lx)#chequear lo de la escala 
vort2=vortF2*(U/Lx)
vort3=vortF3*(U/Lx)

plt.plot(QG_diag1[:,0],QG_diag1[:,3],'c',label='k1')
plt.plot(QG_diag2[:,0],QG_diag2[:,3],'r',label='k2')
plt.plot(QG_diag3[:,0],QG_diag3[:,3],'g',label='k3')
plt.ylabel('Energia cinetica')
plt.xlabel('Tiempo')
plt.legend()

#plt.savefig('Energiacin.png',dpi=100)
plt.show()

#funcion corriente
a=np.min(corriente1)
a=round(a)
levels=np.arange(a,0,a/7)

plt.contourf(X1,Y1,corriente1)
plt.colorbar(levels)
plt.title('Funcion corriente 1')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.xticks(np.arange(0,4000000,1000000))

#plt.savefig('Campo de funcion corriente1.png')
plt.show()

plt.contourf(X2,Y2,corriente2)
plt.colorbar()
plt.title('Funcion corriente 2')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.xticks(np.arange(0,4000000,1000000))

#plt.savefig('Campo de funcion corriente2.png')
plt.show()

plt.contourf(X3,Y3,corriente3)
plt.colorbar()
plt.title('Funcion corriente 3')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.xticks(np.arange(0,4000000,1000000))
#plt.savefig('Campo de funcion corriente3.png')
plt.show()
#transporte meridional

v= np.diff(corriente2,n=1,axis=-1)
D= 2000# profundidad 
my= (Lx/nx)*D*v #transporte meridional
#paso a Sverdrups
my = my/(10**6)
X= np.reshape(X,len(X)-1)
plt.contour(X,Y,my)
plt.colorbar()
plt.xlabel('Transporte meridional')

plt.savefig('Transporte meridional.png')
plt.show()

#parte c corte zonal de transporte meridional
plt.figure()
plt.plot(my[50,:], label='k1')
plt.show()
#ejercicio 2
#ejercicio 3
"""adimensionalizamos rotor del viento, función de corriente y fricción """
rotorviento= QG_curlw2*Tau*-1
friccion= eps3*vortF2*(beta*Lx)
ef=K2/(beta*(Lx**2))
corriente= corriente2*ef
plt.plot(vortF2[50,:],'r',label='rotor de viento')
plt.plot(QG_curlw2[50,:],'m',label='fricción')
plt.plot(v[50,:],'c',label='función corriente')
plt.legend()

plt.show()
#ejercicio 4