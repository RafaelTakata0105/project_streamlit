import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

#Funciones
def yn_var_replace(dataframe, list):
    for x in list:
        dataframe[x] = dataframe[x].replace({'yes': 1, 'no': 0})
def bool_var_replace(dataframe, list):
    for x in list:
        unique_values = dataframe[x].unique()
        dataframe[x] = dataframe[x].replace({unique_values[0] : 1, unique_values[1]: 0})
def concatenate_one_hot(dataframe, list):
    for x in list:
        mom_values = str('Mjob_' + x)
        dad_values = str('Fjob_' + x)
        dataframe[x] = dataframe[mom_values] + dataframe[dad_values]
        dataframe.drop(columns=[mom_values, dad_values], inplace=True)

#Objetos
sts = StandardScaler()
minmax = MinMaxScaler()
labelencoder = LabelEncoder()
tree = DecisionTreeClassifier()
linear = LinearRegression()
lasso = Lasso()
ridge = Ridge()

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

st.markdown('¿Es posible crear un modelo que prediga la calificación que obtendrá el estudiante?')
st.subheader('Dataframes de la limpieza y preparación de los datos')
#Preparación de los datos
yn_var = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
bool_var = ['school', 'sex', 'address', 'famsize', 'Pstatus']
job_values = list(porclass_df['Mjob'].unique())

#Clase de portugues
yn_var_replace(porclass_df, yn_var)
bool_var_replace(porclass_df, bool_var)
porclass_df = pd.get_dummies(porclass_df, columns = ['Mjob', 'Fjob'])
porclass_df['health_status'] = porclass_df['health']
concatenate_one_hot(porclass_df, job_values)
porclass_df['reason'] = labelencoder.fit_transform(porclass_df['reason'])
porclass_df['guardian'] = labelencoder.fit_transform(porclass_df['guardian'])
porclass_df['Pedu'] = porclass_df['Fedu'] + porclass_df['Medu']
porclass_df['alcohol'] = porclass_df['Dalc'] + porclass_df['Walc']
porclass_df['Grade'] = porclass_df['G1'] + porclass_df['G2'] + porclass_df['G3']
porclass_df = porclass_df.drop(columns= ['Fedu', 'Medu', 'Dalc', 'Walc', 'G1', 'G2', 'G3'], axis = 'columns')
porclass_df['Grade'] = pd.cut(porclass_df['Grade'], bins=[-float('inf'), 36, float('inf')], labels=[0, 1], right=False)
porclass_df['age'] = sts.fit_transform(porclass_df['age'].values.reshape(-1, 1))
porclass_df['absences'] = minmax.fit_transform(porclass_df['age'].values.reshape(-1, 1))

st.markdown('Dataframe de la clase de portugues despues de la limpieza:')
st.dataframe(porclass_df)

#Clase de matematicas
yn_var_replace(math_df, yn_var)
bool_var_replace(math_df, bool_var)
math_df = pd.get_dummies(math_df, columns = ['Mjob', 'Fjob'])
math_df['health_status'] = math_df['health']
concatenate_one_hot(math_df, job_values)
math_df['reason'] = labelencoder.fit_transform(math_df['reason'])
math_df['guardian'] = labelencoder.fit_transform(math_df['guardian'])
math_df['Pedu'] = math_df['Fedu'] + math_df['Medu']
math_df['alcohol'] = math_df['Dalc'] + math_df['Walc']
math_df['Grade'] = math_df['G1'] + math_df['G2'] + math_df['G3']
math_df = math_df.drop(columns= ['Fedu', 'Medu', 'Dalc', 'Walc', 'G1', 'G2', 'G3'], axis = 'columns')
math_df['Grade'] = pd.cut(math_df['Grade'], bins=[-float('inf'), 36, float('inf')], labels=[0, 1], right=False)
math_df['age'] = sts.fit_transform(math_df['age'].values.reshape(-1, 1))
math_df['absences'] = minmax.fit_transform(math_df['age'].values.reshape(-1, 1))

st.markdown('Dataframe de la clase de matemáticas despues de la limpieza:')
st.dataframe(math_df)

st.header('Probemos los resultados de diferentes métodos de clasificación')
models = [tree, linear, lasso, ridge]
X = porclass_df.drop(columns = ['Grade'], axis = 1)
y = porclass_df['Grade']
model = st.selectbox('Selecciona las columnas a graficar:', models)
cv_model = cross_val_score(model, X, y, cv=5).mean()
st.write(f'{model}: {cv_model}')