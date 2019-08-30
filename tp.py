# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import numpy as np
from matplotlib import pyplot as plt

dir_salida='C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp3/'
Lx=4000000
Ly=2000000
nx=200
ny=100
beta= 10e-11
from Cargar import cargar
psi_temp,vort_temp,psiF,vortF,QG_diag,QG_curlw,X,Y,dx,dy=cargar(dir_salida,Lx,Ly,nx,ny)
L= Lx
Tau= 0.25
U= (2*np.pi*Tau)/(1025*2000*(beta)*(L))
corriente= psiF*U*L
vort=vortF*(U/L)#chequear lo de la escala 

plt.plot(QG_diag[:,3],'c',label='Energia cinetica')
plt.xlabel('Energia cinetica')
plt.ylabel('Tiempo')
plt.legend()

plt.savefig('Energiacin3.png')
plt.show()

plt.contourf(corriente)
plt.colorbar()
plt.xlabel('Funcion corriente')

plt.savefig('Campo de funcion corriente3.png')
plt.show()

