# Reto MA2008B para Ronal Group

## Introducción

Este proyecto busca modelar la dinámica de horno de sostenimiento de aluminio, con un autoencoder LSTM y la identificación del espacio de estados por medio de DMDc del espacio latente, con el fin de comprarlo con un modelo creado por teoría de control moderno.

## Requisitos e instalación

Se recomienda utilizar python 3.12, para instalar las librerías utilicé el siguiente comando:

```bash
pip install -r requirements.txt
```
Se recomienda crear un entorno virtual, ya sea directamente con `pip` o con `uv`.

## Estandares de desarrollo

### Sistema

Este proyecto fue desarrollado usando software y hardware de Apple, especificame, se utilizó una Macbook Pro, con un chip M1 Pro, 16 GB de ram unificada. Esta base hace que `torch` no este preparado para funcionar en otros dispositivos directamente. Se recomienda ajustar el codígo para las secciones especificamente de entrenamiento de modelo y utilización del mismo.

Actualmente el proyecto esta configurado para ejecutar el modelo directamente sobre el CPU, lo cual dada la densidad de los datos, no es recomendable en muchas maquinas. 

### Reproducibilidad

Este sistema es reproducible y permite correr el pipeline desde 0 con los mismos resultados siempre ya que se configuro la semilla base con el numero `42`. (Nota: Modificar el tipo de ejecución de CPU a GPU puede alterar los resultados incluso con la misma semilla).

### Parametros

Toda la paremetrización existe en un archivo `config.json`.

### Salidas

Todas las salidas generadas por los scripts se pueden encontrar en la ruta: `salidas/`

