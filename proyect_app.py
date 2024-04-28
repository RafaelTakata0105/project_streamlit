import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Proyecto de analisis de Programación de datos')
st.header('Rafael Takata Garcia')
st.subheader('Primavera 2024')

st.markdown('La escuela es una etapa de formación en la que desarrollamos habilidades y conocimientos como ninguna otra. Comprender los factores que afectan al rendimiento de los alumnos nos ayudaría a crear un sistema educativo capaz de maximizar el aprendizaje y aprovechar realmente la educación recibida.')
st.markdown('En 2008, Paulo Cortez y Alice Silva se dedicaron a recoger el mayor número de características de algunos estudiantes de secundaria en el país de Portugal. Estudiaron dos grupos, los que estudiaban portugués y los que estudiaban matemáticas. En este proyecto, trabajaremos con el primero para intentar desarrollar un modelo de predicción para el segundo.')
st.markdown("La creación de un modelo capaz de describir esta relación nos informaría sobre las áreas de oportunidad dentro del sistema educativo de la nación, además de aportar datos significativos sobre la juventud del país.")

st.subheader('Clase de portugués:')
porclass_df = pd.read_csv('student-por.csv')
st.dataframe(porclass_df)

st.subheader('Clase de Matemáticas:')
math_df = pd.read_csv('student-mat.csv')
st.dataframe(math_df)

st.subheader('Haz tus propias gráficas de la clase de portugués:')
graf_col = st.selectbox('Selecciona las columnas a graficar:', porclass_df.columns)
fig, ax = plt.subplots()
ax.hist(porclass_df[graf_col], bins=20, alpha=0.7, color='blue')
ax.set_title(f'Distribución de {graf_col}')
ax.set_xlabel('Valor')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)