# Importacion de las librerias
# Carga del dataset iris
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# Visualizacin del dataset iris
import pandas as pd

iris = load_iris() # Carga del dataset iris
print("")
print("")
print("Vector de las claves del diccionario IRIS:")
print(list(iris.keys())) # Impresion de las claves del diccionario iris
print("")
print("")
print("Vector target con las clases de IRIS:")
print(iris.target) # Impresion de la descripcion del dataset iris

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target # Creacion de una nueva columna llamada species que contiene los valores de la columna target
df['species'] = df['species'].apply(lambda x: iris.target_names[x]) # Cambio de los valores de la columna species por los nombres de las especies
print("")
print("")
print("Primeras 5 filas del dataset IRIS:")
print(df.head()) # Impresion de las primeras 5 filas del dataset


# visualización mediante un gráfico de dispersión

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()

# Creacion del objeto Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)] # Seleccion de la longitud y ancho del petalo.
y = (iris.target == 0) #  Crea el vector objetivo y, que será True si la flor es de la especie Iris Setosa (clase 0) y False en caso contrario.
print("")
print("")
print("Impresión del vetor y:")
print(y) # Impresion del vector objetivo

per_clf = Perceptron() # Creacion del objeto Perceptron
per_clf.fit(X, y) # entrenamiento del modelo

y_pred = per_clf.predict([[1, 0.5]]) # prediccion de estas dos flores en verdadero y y falso
print("")
print("")
print("Impresión de la prediccion")
print(y_pred) # Impresion de la predicción

# Función para crear una malla de puntos
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = per_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.xlabel("Longitud del pétalo")
    plt.ylabel("Ancho del pétalo")
    plt.title("Perceptrón - Frontera de decisión")
    plt.show()

# Graficar la frontera de decisión
plot_decision_boundary(per_clf, X, y)