
from dataclasses import dataclass
from sklearn.utils import column_or_1d
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns   

def EDA(Data):
    Datos = pd.read_csv(Data)
    st.subheader('Muestra de datos')
    st.dataframe(Datos)
    col1, col2= st.columns(2)

    with col1:

        st.write("1.-Descripción de la estructura de los datos ")
        st.write("**Dimensión**")
        st.text('Renglones:\t'+ str(Datos.shape[0]))
        st.text('Columnas:\t'+ str(Datos.shape[1]))

    with col2:
        st.write("Tipos de datos")
        st.text(Datos.dtypes)
        

    st.subheader("2.-Identificación de los datos faltantes")
    st.write("Total de Valores Nulos")
    st.text(Datos.isnull().sum())

    st.write("Información de los datos")
    st.text(Datos.info)
      
    

    st.subheader("3.-Detección de Datos Atípicos")
    st.subheader("3.1.-Distribución de variables númericas.")
    st.text("Se pueden utilizar gráficos para tener una idea general de las distribuciones de los datos, y se sacan estadísticas para resumir los datos. Estas dos estrategias son recomendables y se complementan.")
    st.text("La distribución se refiere a cómo se distribuyen los valores en una variable o con qué frecuencia ocurren.")
    st.text("Para las variables numéricas, se observa cuántas veces aparecen grupos de números en una columna. Mientras que para las variables categóricas, son las clases de cada columna y su frecuencia")

    Datos.hist(figsize=(14,14), xrot=45)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    

    Columna = st.text_input('Digita la Columna de análisis tal como aparece en el Dataframe ')
    if Columna:
            Datos[Datos.country == Columna]
            st.dataframe(Datos[Datos.country == Columna])

            Datos[Datos.country == Columna].hist(figsize=(14,14), xrot=45)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
      
            st.write("3.2.-Resumen estadístico de variables númericas")
            Datos[Datos.country == Columna].describe()
            st.dataframe(Datos[Datos.country == Columna].describe())

       #ARREGLAR
            st.write("3.3.-Diagramas  para detectar posibles valores atípicos")
            Columna1 = st.text_input('Ingresar variables atipicas ')
            if Columna1:
                    VariablesValoresAtipicos = [Columna1]
                    for col in VariablesValoresAtipicos:
                        sns.boxplot(col, data=Datos[Datos.country == Columna])
                        plt.show()
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()

        
            st.write("4.-Identificación de relaciones entre pares variables")
            Datos[Datos.country == Columna].corr()
            st.dataframe(Datos[Datos.country == Columna].corr())
            
     
            plt.figure(figsize=(14,7))
            sns.heatmap(Datos[Datos.country == Columna].corr(), cmap='RdBu_r', annot=True)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(Datos[Datos.country == Columna].corr())
            sns.heatmap(Datos[Datos.country == Columna].corr(),cmap='RdBu_r',annot=True,mask=MatrizInf)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
