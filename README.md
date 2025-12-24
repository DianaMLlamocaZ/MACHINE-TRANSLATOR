# Machine Translator (Inglés -> Español)
Este repositorio contiene un traductor de inglés a español desarrollado íntegramente desde cero, implementando la arquitectura Seq2Seq con el mecanismo de atención Bahdanau y GRU layers.

---

## Vista previa
Aquí muestro una captura de pantalla de prueba del modelo mediante la interfaz.
![TestModelo](./imgs/Test%Modelo.JPG)

---
## Dataset:
El modelo fue entrenado utilizando el dataset **English-Spanish Translation Dataset** alojado en Kaggle.

## Arquitectura del modelo
El modelo traductor ha sido implementado desde cero, enfocándome principalmente en el procesamiento correcto de tensores y su arquitectura.
- **Encoder:** Contiene 2 GRU layers que codifican la oración en hidden state vectors que se generan en cada paso a tiempo.
- **Attention:** Implementación de **Bahdanau Attention** desde cero, que se encarga de generar un vector de contexto que le otorga diferentes pesos a los hidden states generados por el encoder. 
- **Decoder:** Toma el context vector (generado por el mecanismo de atención), el input actual y el hidden state para predecir las palabras en el idioma objetivo.
- **Embedding Layer:** Convierte el id token (representación de cada palabra en el vocabulario) a vectores de 'n' dimensiones (word embedding), n: 512.

NOTA: Son 2 modelos que se prueban en la interfaz y tienen los mismos hiperparámetros, solo que el 'modelo 2' fue entrenado con más épocas.

---

## Características principales
- **Pipeline de datos:** Limpieza, normalización y tokenización del dataset.
- **Inferencia en tiempo real:** Cuando el usuario coloca una oración en inglés, se realiza la traducción a español en tiempo real y se muestra en la interfaz.
- **Interfaz Streamlit:** Permite al usuario interactuar con el modelo de forma directa.

---

## Información adicional importante
- Para cada idioma, creé 1 diccionario que mapea las palabras de ese idioma a un id token, necesario para generar, a partir de él, el word embedding a través del Embedding Layer.
- Para que el entrenamiento del modelo sea eficiente y paralelo, utilicé batches. Por ello, debido a que no todas las oraciones tenían el mismo tamaño durante el entrenamiento, utilicé la técnica de 'padding'.
- Luego de entrenar al modelo con diferentes hiperparámetros, la siguiente configuración resultó en un mejor rendimiento:
  - a
  - b
  - c
---

## Instalación y uso local
Para ejecutar el programa de forma local, sigue los pasos descritos:

### **Ubuntu**
- Clonar el repositorio (recomendado en el escritorio):
```bash
git clone (link_del_repo) machine-translator
```
- Entrar a la carpeta donde clonaste el repositorio:
```bash
cd [Ruta_donde_clonaste_el_repositorio]
```
- Crear el environment (en esa misma carpeta):
```bash
python3 -m venv machine-translator-env
```
- Activar el environment:
```bash
source machine-translator-env/bin/activate
```
- Instalar las librerías necesarias:
```bash
pip install -r requirements.txt
``` 
- Ejecutar la aplicación:
```bash
streamlit run interfaz.py
``` 

### **Windows**
- Clonar el repositorio (recomendado en el escritorio):
```bash
git clone (link_del_repo) machine-translator
```
- Entrar a la carpeta donde clonaste el repositorio:
```bash
cd [Ruta_donde_clonaste_el_repositorio]
```
- Crear el environment (en esa misma carpeta):
```bash
python -m venv machine-translator-env
```
- Activar el environment:
```bash
.\machine-translator-env\Scripts\activate
```
- Instalar las librerías necesarias:
```bash
pip install -r requirements.txt
``` 
- Ejecutar la aplicación:
```bash
streamlit run interfaz.py
``` 
