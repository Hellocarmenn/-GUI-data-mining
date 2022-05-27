from click import option
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree


def clasificacion(Data):
    Data = pd.read_csv(Data)
    st.subheader('Muestra de datos')
    st.dataframe(Data)
    optionsx = st.multiselect(
        'Seleccionar  las variables predictoras (X) ',
        Data.columns, key = 0
    )
    st.write("Las variables predictoras se seleccionan  una vez ya realizacion la seleccion de caracteristicas")
    if optionsx != []:
        X = np.array(Data[optionsx])
        optionsy = st.multiselect(
        'Selecciona la columna para la variable clase(Y)',
        Data.columns, key = 1)
        if optionsy != []:
            Y = np.array(Data[optionsy])
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)
            option = st.selectbox(
            ' Seleccionar un  tipo de Algoritmo de clasificación',
            ('','Árbol de decisión: Clasificación','Bosque Aleatorio: Clasificación')
            )
        if option == 'Árbol de decisión: Clasificación':
            Arbol_Clasificacion(optionsx,X_train, X_test, Y_train, Y_test)
        elif option == 'Bosque Aleatorio: Clasificación':
            Bosque_Clasificacion(optionsx,X_train, X_test, Y_train, Y_test)


    def Arbol_Clasificacion(optionsx, X_train, X_test, Y_train, Y_test):
        ClasificacionAD = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2)
        ClasificacionAD.fit(X_train, Y_train)
        Y_Clasificacion = ClasificacionAD.predict(X_test)
        st.write("Reporte de la Clasificación")
        st.write('Criterio: \n', ClasificacionAD.criterion)
        st.write('Importancia variables: \n', ClasificacionAD.feature_importances_)
        st.write("Bondad de Ajuste", ClasificacionAD.score(X_test, Y_test))
        st.text(classification_report(Y_test, Y_Clasificacion))
        st.write("Matriz de Clasificación")
        Matriz_Clasificacion = pd.crosstab(Y_test.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['Real'], 
                                            colnames=['Clasificación']) 
        st.dataframe(Matriz_Clasificacion)
        st.write("Gráfico del Bosque Generado")
        plt.figure(figsize=(16,16))  
        plot_tree(ClasificacionAD, 
                feature_names = optionsx,
                class_names = Y_Clasificacion)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    def Bosque_Clasificacion(optionsx,X_train, X_test, Y_train, Y_test):
        st.subheader("Bosque Aleatorio: Clasificación")
        ClasificacionBA = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
        ClasificacionBA.fit(X_train, Y_train)
        Y_Clasificacion = ClasificacionBA.predict(X_test)
        ClasificacionBA.score(X_test, Y_test)
        st.write("Reporte de la Clasificación")
        st.write('Criterio: \n', ClasificacionBA.criterion)
        st.write('Importancia variables: \n', ClasificacionBA.feature_importances_)
        st.write("Bondad de Ajuste", ClasificacionBA.score(X_test, Y_test))
        st.text(classification_report(Y_test, Y_Clasificacion))
        st.write("Matriz de Clasificación")
        Matriz_Clasificacion = pd.crosstab(Y_test.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['Real'], 
                                            colnames=['Clasificación']) 
        st.dataframe(Matriz_Clasificacion)
        st.write("Gráfico del Bosque Generado")
        number = st.number_input('Digita el Número de estimadores')
        Estimador = ClasificacionBA.estimators_[int(number)]
        plt.figure(figsize=(16,16))  
        plot_tree(Estimador, 
                    feature_names = optionsx,
                    class_names = Y_Clasificacion)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


