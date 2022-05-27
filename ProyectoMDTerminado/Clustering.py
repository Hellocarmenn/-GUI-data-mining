import streamlit as st
import pandas as pd
import numpy as np   
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min 
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D



def clutering(Data):
     
    Data=pd.read_csv(Data)
    st.subheader('Muestra de datos')
    st.dataframe(Data)

    option = st.selectbox(
      ' Seleccionar un tipo de Clúster',
      ('Clúster Jerárquico','Clúster Particional')
    )
    if option == 'Clúster Jerárquico':
      jeraquico(Data)
    elif option == 'Clúster Particional':
      particional(Data)
     
    
def jeraquico(Data):
  st.write("Una vez realizado la seleción de caracteristicas, ingresar las variables a eleminar")
  st.write("Si no identifica las variables a eliminar puede acceder previamente a la pestaña 'Selección de Características' donde encontrá dos formas de seleccionar que variables son necesarias para su análisis")
  options = st.multiselect(
      'Selecciona las columnas que serán eliminadas tal como aparece en el Datosñ',
      Data.columns, key = 0
    )
  col1, col2 = st.columns(2)
  with col1:      
    st.write('Columnas Eliminadas:', options)
    elimination = Data.drop(columns=options)      
  with col2:
    st.write('Columnas Restantes:', elimination.columns.tolist())
  #st.dataframe(elimination)
  MEstandar=estandarizacion(elimination)
  if  MEstandar is not None:
    st.dataframe(MEstandar)
    plt.figure(figsize=(10, 7))
    plt.title("Casos del árbol Gerárquico")
    plt.xlabel('Componentes')
    plt.ylabel('Distancia')
    option = st.selectbox(
      ' Elige un tipo de Metrica de Distancia',
      ('Euclidiana','Chebyshev', 'Manhattan'))

    if option == 'Euclidiana':
      Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='euclidean'))
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
      
      number=st.number_input("Selecciona el número de Clústers respecto al gráfico")
      if number == 0:
        st.write("Selecciona un valor diferente de 0")
      else:
        st.subheader("**Cantidad de Clústers**")
        MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='euclidean')
        MJerarquico.fit_predict(MEstandar)
        elimination.assign(Cluster_H = 0)
        elimination['Clúster_H'] = MJerarquico.labels_
        CentroidesH = elimination.groupby('Clúster_H').mean()
        st.dataframe(CentroidesH)
        st.subheader("**Elementos de los Clústers**")
        plt.figure(figsize=(10, 7))
        plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
        plt.grid()
        plt.show() 
        st.pyplot()

    elif option == 'Chebyshev':
      Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='chebyshev'))
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
      number=st.number_input("Selecciona el número de Clústers respecto al gráfico, es decir respecto al numero de colores que aparece en el grafico")
      if number == 0:
        st.write("Seleccionar un valor diferente de 0")
      else:
        st.subheader("Cantidad de Clústers")
        MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='chebyshev')
        MJerarquico.fit_predict(MEstandar)
        elimination.assign(Cluster_H = 0)
        elimination['Clúster_H'] = MJerarquico.labels_
        CentroidesH = elimination.groupby('Clúster_H').mean()
        st.dataframe(CentroidesH)
        st.subheader("Elementos de los Clústers")
        plt.figure(figsize=(10, 7))
        plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
        plt.grid()
        plt.show() 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    elif option == 'Manhattan':
      Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='cityblock'))
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
      

      number=st.number_input("Selecciona el número de Clústers respecto al gráfico")
      if number == 0:
        st.write("Seleccionar un valor diferente de 0")
      else:
        st.subheader("Cantidad de Clústers")
        MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='cityblock')
        MJerarquico.fit_predict(MEstandar)
        elimination.assign(Cluster_H = 0)
        elimination['Clúster_H'] = MJerarquico.labels_
        CentroidesH = elimination.groupby('Clúster_H').mean()
        st.dataframe(CentroidesH)
        st.subheader("Elementos de los Clústers")
        plt.figure(figsize=(10, 7))
        plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
        plt.grid()
        plt.show() 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
  else:
    st.write("Es necesario seleccionarun metodo de estandarización para continuar con el algortimo")

def estandarizacion (Data):
  sel = st.selectbox('Seleccionar una forma de estandarización de datos',
            ('','Estandarización', 'Normalización'))
  if sel == 'Estandarización':
    st.subheader('Estandarización')
    Estandarizar = StandardScaler()
    MEstandar = Estandarizar.fit_transform(Data)
    return MEstandar  
  elif sel == 'Normalización':
    st.subheader('Normalización')
    Normalizar = MinMaxScaler() 
    MNormalizar = Normalizar.fit_transform(Data) 
    return MNormalizar

def particional(Data):
  st.write("Una vez realizado la seleción de caracteristicas, ingresar las variables a eleminar")
  st.write("Si no identifica las variables a eliminar puede acceder previamente a la pestaña 'Selección de Características' donde encontrá dos formas de seleccionar que variables son necesarias para su análisis")
  options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 0
    )
  col1, col2 = st.columns(2)
  with col1:      
    st.write('Columnas Eliminadas:', options)
    elimination = Data.drop(columns=options)      
  with col2:
    st.write('Columnas Restantes:', elimination.columns.tolist())
  st.dataframe(elimination)
  MEstandar=estandarizacion(elimination)
  if  MEstandar is not None:
    SSE = []
    for i in range(2, 12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandar)
        SSE.append(km.inertia_)

    st.subheader(" Cantidad de Clústers por el algortimo k-means")
    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
    cluster=kl.elbow
    plt.style.use('ggplot')
    kl.plot_knee()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    MParticional = KMeans(n_clusters = cluster, random_state=0).fit(MEstandar)
    MParticional.predict(MEstandar)
    MParticional.labels_




    Data['clusterP'] = MParticional.labels_
    CentroidesP = Data.groupby('clusterP').mean()
    st.subheader("**Clústers Asociados**")
    st.dataframe(CentroidesP) 

    st.subheader("Grafica de elementos y centros de Clústers")
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.style.use('ggplot')
    colores=['red', 'blue', 'green', 'yellow']
    asignar=[]
    for row in MParticional.labels_:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(MEstandar[:, 0], 
              MEstandar[:, 1], 
              MEstandar[:, 2], marker='o', c=asignar, s=60)
    ax.scatter(MParticional.cluster_centers_[:, 0], 
              MParticional.cluster_centers_[:, 1], 
              MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)