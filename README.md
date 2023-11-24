Este código es un ejemplo de una aplicación web utilizando el framework Flask en Python. La aplicación tiene un modelo de red neuronal (en este caso, un modelo simple de clasificación binaria) que puede predecir el riesgo de ataque al corazón basándose en ciertos datos médicos. Además, la aplicación permite realizar predicciones en tiempo real a través de la API REST.

Aquí hay un resumen de cómo funciona la aplicación:

### Configuración y Middleware:

Se importan las bibliotecas necesarias, como Flask, PyODBC, y otras.
Se define un middleware check_credentials para verificar las claves API y el nombre del proyecto en las solicitudes entrantes.

### Conexión a la Base de Datos:

Se establece una conexión a una base de datos SQL Server utilizando PyODBC.

###Preprocesamiento de Datos:

Los datos se cargan desde un archivo CSV y se preprocesan para su uso en el modelo de red neuronal. Esto incluye la eliminación de columnas innecesarias, la manipulación de la columna 'Blood Pressure', el manejo de valores faltantes, la codificación one-hot de variables categóricas, y la normalización de características.

### Entrenamiento del Modelo:

Se define un modelo de red neuronal (SimpleNN) utilizando PyTorch.
Los datos se dividen en conjuntos de entrenamiento y prueba.
Se entrena el modelo durante un número específico de épocas.
Endpoints de la API:

/predict (método POST): Se utiliza para hacer predicciones con nuevos datos.
Los datos de formulario JSON se procesan y preprocesan de manera similar a los datos de entrenamiento.
Se realiza una predicción con el modelo entrenado.
El resultado se inserta en la base de datos, y el modelo se actualiza con los nuevos datos.
La respuesta contiene un ID de transacción en lugar de la probabilidad de predicción.
/obtener_prediccion/<transaction_id> (método GET): Se utiliza para obtener información de predicciones anteriores.
Se consulta la base de datos utilizando el ID de transacción.
La respuesta contiene la probabilidad de predicción y otra información asociada.

### Servidor Web:

La aplicación se inicia utilizando el servidor Waitress y se ejecuta en el puerto 5000.
Para ejecutar esta aplicación, debes asegurarte de tener todas las dependencias instaladas (Flask, Waitress, PyTorch, etc.). Puedes instalar las dependencias ejecutando pip install Flask waitress torch pandas scikit-learn pyodbc.

Luego, ejecutas el script Python y la aplicación estará disponible en http://127.0.0.1:5000/. Puedes hacer solicitudes a los endpoints /predict y /obtener_prediccion/<transaction_id> utilizando herramientas como cURL, Postman, o desde otro script Python.

Recordar que para correr La aplicacion debe estar conectada a una base de datos local.

Query 

-- Crear la base de datos pruebas_servicios
CREATE DATABASE pruebas_servicios;
GO

-- Cambiar al contexto de la base de datos pruebas_servicios
USE pruebas_servicios;
GO

-- Crear una tabla para almacenar las predicciones
CREATE TABLE Predictions (
    TransactionID NVARCHAR(255) PRIMARY KEY,
    Probability FLOAT,
    OtherInfo NVARCHAR(255)
);
GO
