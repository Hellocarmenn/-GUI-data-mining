import csv
import streamlit as st
import pandas as pd
from funciones import Switch


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


Title ="""
<h1><center>Universida Nacional Autónoma de México</center></h1>
<h3><center>Mineria de Datos</center></h3>
<h4><center>Martinez Salas Maria Del Carmen</center></h4>
<h5><center>carmenmtz97@comunidad.unam.mx</center></h5>
<br>
"""
st.markdown(Title ,unsafe_allow_html=True) #informacion principal
st.text('Inserta archivo con extensión CSV')
data = st.file_uploader('Ingresa tu archivo', type =['csv']) #insercion de datos tipo csv de forma local

if data is not None:
   ## dataframe = pd.read_csv(data)
    dataframe=data
    Algoritmo = st.selectbox('Elige un algoritmo de implementacion',
                ('Muestra de Datos','Análisis Exploratorio de Datos', 'Selección de características',  
                'Clustering', 
                'Reglas de Asociación', 
                'Pronostico',
                'Clasificacion'))
    Switch(Algoritmo, dataframe)