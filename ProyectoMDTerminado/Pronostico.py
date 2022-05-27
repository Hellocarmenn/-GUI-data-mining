import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

def pronostico(Data):

    Data = pd.read_csv(Data)
    st.subheader('Muestra de datos')
    st.dataframe(Data)
         
    optionsx = st.multiselect(
        'Seleccionar  las columnas para las variables predictoras (X)',
        Data.columns, key = 0
    )
    st.write("Las variables predictoras se seleccionan  una vez ya realizacion la seleccion de caracteristicas")
    if optionsx != []:
        X = np.array(Data[optionsx])
        optionsy = st.multiselect(
        'Seleccionar la columna para la variable clase(Y)',
        Data.columns, key = 1)
        if optionsy != []:
            Y = np.array(Data[optionsy])
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)
            option = st.selectbox(
            ' Un algoritmo para Pronóstico',
            ('','Árbol de decisión: Pronóstico','Bosque Aleatorio: Pronóstico')
            )
        if option == 'Árbol de decisión: Pronóstico':
            Arbol_Pronostico(optionsx, X_train, X_test, Y_train, Y_test)
        elif option == 'Bosque Aleatorio: Pronóstico':
            Bosque_Regresion(X_train, X_test, Y_train, Y_test)

    def Arbol_Pronostico(optionsx,X_train, X_test, Y_train, Y_test):
        PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
        PronosticoAD.fit(X_train, Y_train)
        Y_Pronostico = PronosticoAD.predict(X_test)
        st.write("Parámetros del Modelo")
        st.write("Bondad de Ajuste", r2_score(Y_test, Y_Pronostico))
        st.write('Criterio: \n', PronosticoAD.criterion)
        st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
        st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
        st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
        st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
        st.write("Conformación del modelo de pronóstico")
        plt.figure(figsize=(16,16))  
        plot_tree(PronosticoAD, feature_names = optionsx)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def Bosque_Regresion(X_train, X_test, Y_train, Y_test):
        st.subheader("Bosque Aleatorio: Regresión")
        PronosticoBA = RandomForestRegressor(random_state=0, max_depth=8, min_samples_leaf=2, min_samples_split=4)
        PronosticoBA.fit(X_train, Y_train)
        Y_Pronostico = PronosticoBA.predict(X_test)
        st.write("Parámetros del Modelo")
        st.write("Bondad de Ajuste", r2_score(Y_test, Y_Pronostico))
        st.write('Criterio: \n', PronosticoBA.criterion)
        st.write('Importancia variables: \n', PronosticoBA.feature_importances_)
        st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
        st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
        st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
