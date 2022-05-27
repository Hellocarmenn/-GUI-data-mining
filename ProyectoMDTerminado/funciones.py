from this import d
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns
from Clasificacion import clasificacion  
from EDA import EDA
from Pronostico import pronostico
from RDA import rda      
from Seleccion_de_caracteristicas import  seleccion_de_caracteristicas          
from Clustering import clutering

def Switch(Election,Data):
    if Election == 'Muestra de Datos':
        st.header("Datos")
        Data=pd.read_csv(Data)
        st.dataframe(Data)
    elif Election == 'Análisis Exploratorio de Datos':
        st.header("Análisis Exploratorio de Datos")
        EDA(Data)
    elif Election == 'Selección de características':
        st.header("Selección de Características")
        seleccion_de_caracteristicas(Data)
    elif Election == 'Clustering':
        st.header("Clústers")
        clutering(Data)
    elif Election == 'Reglas de Asociación': 
        st.header('Reglas de Asociación')
        rda(Data)
        print("hola")
    elif Election == 'Pronostico':
        st.header('Pronostico')
        pronostico(Data)
    elif Election == 'Clasificacion':
        st.text('Clasificacion')
        clasificacion(Data)



    

