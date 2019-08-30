# -*- coding: utf-8 -*-
"""
Practica 1 - Circulacion
grupo: Micaela; Luciano.
"""

import numpy as np
from matplotlib import pyplot as plt

# SI HAY ALGUN PROBLEMA CON LOS VALORES, ES DE ESCALA NO DE UNIDADES.
# usamos metros para calcular beta --> eps por lo tanto usamos TODO en metros.

Lx = 4000000 # dimenciones de la cuenca en metros
Ly = 2000000
nx = 200 # puntos de grilla
ny = 100
beta = 10**-11 # en metros
D = 2000 # metros

#arrays (y,x) !

from Cargar import cargar 
psi_temp1,vort_temp1,psiF1,vortF1,QG_diag1,QG_curlw1,X1,Y1,dx1,dy1=cargar('C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp1/',Lx,Ly,nx,ny)
psi_temp2,vort_temp2,psiF2,vortF2,QG_diag2,QG_curlw2,X2,Y2,dx2,dy2=cargar('C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp2/',Lx,Ly,nx,ny)
psi_temp3,vort_temp3,psiF3,vortF3,QG_diag3,QG_curlw3,X3,Y3,dx3,dy3=cargar('C:/Users/Jimena/Desktop/FCEN/circulación/prácticas/TP1/out_tmp3/',Lx,Ly,nx,ny)

###----------------1----------------###

# dimensionalizando 
Tau = 0.25
U = (2*np.pi*Tau)/(1025*D*(beta)*(Lx))

corriente1 = psiF1*U*Lx
corriente2 = psiF2*U*Lx
corriente3 = psiF3*U*Lx

vort1 = vortF1*(U/Lx)
vort2 = vortF2*(U/Lx)
vort3 = vortF3*(U/Lx)

# Energia cinetica. (no la dimensionalizamos)
plt.figure()
plt.plot(QG_diag1[:,3],'r',label='K1')
plt.plot(QG_diag2[:,3],'g',label="K2")
plt.plot(QG_diag3[:,3],'b',label="K3")
plt.title("Energia Cinética")
plt.xlabel("Tiempo")
plt.ylabel("Energia cinética")
plt.legend()
plt.tight_layout()
plt.savefig("energia_cin.png",dpi=200)


# Funcion corriente
escala = np.arange(-420000,60000,70000)
corrientes = (corriente1,corriente2,corriente3)
nombres_c = ("corriente1","corriente2","corriente3")
titulo_c = ("Corriente K1", "Corriente K2", "Corriente K3")
num = (0,1,2)

for i in num:
    plt.figure()
    plt.contourf(X1,Y1,corrientes[i],escala,cmap='winter')
    plt.colorbar()
    plt.title(titulo_c[i])
    plt.xlabel('longitud')
    plt.ylabel('latitud')
    plt.xticks(np.arange(0,4000000,1000000))
    plt.tight_layout()
    plt.savefig(nombres_c[i],dpi=200)
    
# transporte meridional, en Sverdrups
my1 = D*Lx/(nx)*(np.diff(corriente1,n=1,axis=1))/(10**6)
my2 = D*Lx/(nx)*(np.diff(corriente2,n=1,axis=1))/(10**6)
my3 = D*Lx/(nx)*(np.diff(corriente3,n=1,axis=1))/(10**6)

# campos de transporte meridional
escala_my = np.arange(-3200000,800000,400000) #escala para los graficos 
# --> no es mucho tres millones doscientos mil Sv???
my = (my1,my2,my3)
nombres = ("ej1_my1","ej1_my2","ej1_my3")
titulo= ("My K1","My K2", "My K3")

for x in num:
    plt.figure()                    
    plt.contourf(X1[0:199],Y1,my[x],escala_my,cmap='winter')
    plt.title(titulo[x])
    plt.colorbar()
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.xticks(np.arange(0,4000000,1000000))
    plt.tight_layout()
    plt.savefig(nombres[x],dpi=200)

# corte zonal en la latitud central de la cuenca
#-->y=50 ESTA BIEN??? my[:,50] o[[ my1[50,:] ]], si! arrays de cargar.py (y,x)

plt.figure()
plt.plot(my1[50,:],'r',label="K1")
plt.plot(my2[50,:],'g',label="K2")
plt.plot(my3[50,:],'b',label="K3")
plt.ylabel("My")
plt.xlabel("km")
plt.title("Transporte meridional")
plt.legend()
plt.tight_layout()
plt.savefig("transporte_meridional", dpi=200)


# lo mismo con la vorticidad
plt.figure()
plt.plot(X1[0:200]/1000,vort1[50,:],'r', label = "K1") #dividido 1000 para que de en km
plt.plot(X1[0:200]/1000,vort2[50,:],'g', label = "K2") 
plt.plot(X1[0:200]/1000,vort3[50,:],'b', label = "K3") 
plt.xlabel("Km")
plt.title("Vorticidad relativa")
plt.ylabel("vort. relativa")
plt.legend()
plt.tight_layout()
plt.savefig("vorticidad_relativa", dpi=200)

###----------------2----------------###
# my pero de la cbo
# buscado a mano... my[50,:], cambia de signo en 22 <-- extencion de la cbo
# esto tambien funca ---> np.where(my1[50,:]==0) 
# hay q buscar donde cambia de signo, puede q no existan ceros ya q no es continuo
# EN TODOS HAY CEROS.

my_cbo1=my1[50,:] # latitud (y) primera coordenada del array numero 50 (centro de la cuenca)
my_cbo1_F=np.sum(my_cbo1[0:22])
my_cbo1_total=np.sum(my_cbo1) # esto ya estaba en sv
extension_cbo1=X1[22]

np.where(my2[50,:]==0)  #---> dos posiciones 46 y 49, viendo la matriz tiene valor 0 -algo + ese mismo algo 0, elijo 49
my_cbo2=my2[50,:]
my_cbo2_F=np.sum(my_cbo2[0:49])
my_cbo2_total=np.sum(my_cbo2) 
extension_cbo2=X2[49,]

np.where(my3[50,:]==0)  #---> varias posiciones, eijo la ultima 66 \m/
my_cbo3=my3[50,:] 
my_cbo3_F=np.sum(my_cbo3[0:66])
my_cbo3_total=np.sum(my_cbo3) 
extension_cbo3=X3[66,]

# hay q presentarlo como tabla --> guardamos en excel usando:
# import pandas
# ordenando 
# variable.to_excel()

import pandas as pd

# se crea un archivo tipo dict ( "dictionary" ~ lista en R )
# solo rendondeo del total ya que da con ordenes 10^-7 pero no cero
# el transporte de borde oste no hace falta

ej_2 = {" " : ["My borde oeste","My total","Extension cbo"],
        "K1":[(my_cbo1_F),round(my_cbo1_total),extension_cbo1],
        "K2":[(my_cbo2_F),round(my_cbo2_total),extension_cbo2],
        "K3":[(my_cbo3_F),round(my_cbo2_total),extension_cbo3]}

# ahora pasado a un dataframe, los caracteres pasan a las filas y columnas
ej_2 = pd.DataFrame(data=ej_2)
ej_2.to_excel("ej_2.xls",index = False)

###----------------3----------------###

## elegimos la simulacion 2 ##
# en forma adimencional 
# en la latitud central de la cuenca ---> [50,:]

# primer termino - derivada en x de la funcion corriente psiF, ya que es cuando se alcanza un estado estacionario--> final
# misma duda que antes, el axis determina la varible en la cual se esta derivando
termino1 = ((np.diff(psiF2,n=1, axis=1)))[50,:]*25 # o /0.05 --> ds del .dat grid step (no entiendo xq, pero sino quedan valores muy chicos en modulo) 

# segundo termino, menos rotor del viento (-QG_curlw )
termino2 = -QG_curlw2[50,:]

# tercer termino
termino3 = 0.79*vortF2[50,:]  # 0.79 valor del eps 2

plt.figure()
plt.plot(termino1,"c",label = "Término de Transporte")
plt.plot(termino2,"r",label = "Término del Rotor de viento")
plt.plot(termino3,"m",label = "Término de fricción")
plt.axhline(y = 0, color = "black") # marco la linea de y = 0
plt.xlabel("X")
plt.ylabel("valores adimensionales")
plt.legend()
plt.tight_layout()
plt.savefig("ej_3.png",dpi = 200)
#veamos que da cero o cercano a cero
#promedio cada termino
prom1=np.mean(termino1)
prom2=np.mean(termino2)
prom3=np.mean(termino3)

stommel=prom1+prom2+prom3
error = (abs(0 -stommel))*100
print(f'El error asociado es {error} %')
 ### LAS UNIDADES QUEDAN RARAS, MUY GRANDES, RESPETAMOS TOMAR TODO EN METROS
### DESDE EL CALCULO DE BETA, LOS VALORES INGRESADOS EN LA FUNCION CAGAR Y LAS CONVERSIONES

###----------------\m/----------------###