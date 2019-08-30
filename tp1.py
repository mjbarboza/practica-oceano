# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:21:46 2019

@author: Jimena
"""

import numpy as np
from Cargar import cargar

K1 = 1.16e-5
K2 = 3.16e-5
K3 = 5.16e-5
beta= 10e-11
L=4000000
eps1= K1/(beta*L)#
eps2= K2/(beta*L)#es un orden menos
eps3= K3/(beta*L)# 
dir_salida= 'C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp1/'
Lx = 4000000
Ly = 2000000
nx = 200
ny = 100
psi_temp,vort_temp,psiF,vortF,QG_diag,QG_curlw,X,Y,dx,dy=cargar(dir_salida,Lx,Ly,nx,ny)
