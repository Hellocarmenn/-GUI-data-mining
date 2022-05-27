from asyncio import PriorityQueue
import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


def rda(Data):
    Datos = pd.read_csv(Data)
    
    st.dataframe(Datos)
    ##Datos = pd.read_csv(Data,header=None)
    
    st.subheader("Procesamiento de los datos")
    st.write("Exploración:Antes de ejecutar el algoritmo  es recomendable observar la distribución de la frecuencia de los elementos.")
    Transacciones = Datos.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
    

    st.write("Se crea una matriz  usando la lista y se incluye una columna 'Frecuencia'")
    ListaM = pd.DataFrame(Transacciones)
    st.dataframe(ListaM)

    ListaM['Frecuencia'] = 0
    st.dataframe(ListaM)

    st.header("Se agrupa los elementos")
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
    ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
    ListaM = ListaM.rename(columns={0 : 'Item'})
    st.dataframe(ListaM)

    st.header("Se genera un grafico de barras")
    fig,ax=plt.subplots()
   # plt.figure(figsize=(12,15), dpi=300)
   # ax.set_ylabel(fontsize=14)
    ax.tick_params(labelsize=3)
    

    ax.set_ylabel('Item')
    ax.set_xlabel('Frecuencia')
    ax.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
   ## ax.show()
    st.pyplot(fig)


    st.header("Se crea una lista de listas a partir del dataframe y se remueven los 'NAN'")
    Lista = Datos.stack().groupby(level=0).apply(list).tolist()
    st.dataframe(Lista)

    ReglasC1 = apriori(Lista, min_support=0.01,  min_confidence=0.3,  min_lift=2)


    st.header("Total de reglas encontradas")
    ResultadosC1 = list(ReglasC1)
    st.write(len(ResultadosC1)) 
   # st.text(ResultadosC1)
    #st.text(pd.DataFrame(ResultadosC1))


    st.header("Impresión de la primera regla")
    st.markdown(ResultadosC1[0])
    


    st.header("Resusltados por indice")
    for item in ResultadosC1:
        Emparejar = item[0]
        items = [x for x in Emparejar]
        st.text("Regla: " + str(item[0]))
        st.text("Soporte: " + str(item[1]))
        st.text("Confianza: " + str(item[2][0][2]))
        st.text("Lift: " + str(item[2][0][3])) 
        st.write("-------------------------------------")















