# Machine Translator (Inglés -> Español)
Este repositorio contiene un traductor de inglés a español desarrollado íntegramente desde cero, implementando la arquitectura Seq2Seq con el mecanismo de atención Bahdanau y GRU layers.

---

## Vista previa
Aquí mostraré un gif de prueba del modelo.

---

## Arquitectura del modelo
El modelo traductor ha sido implementado desde cero, enfocándome principalmente en el procesamiento correcto de tensores y su arquitectura.
- **Encoder:** Contiene 2 GRU layers que codifican la oración en hidden state vectors que se generan en cada paso a tiempo.
- **Attention:** Implementación de **Bahdanau Attention** desde cero, que genera un vector de contexto que le otorga diferentes pesos a los hidden states generados por el encoder. 
- **Decoder:** Toma el context vector (generado por el mecanismo de atención), el input actual y el hidden state para predecir las palabras en el idioma objetivo.
- **Embedding Layer:** Convierte el input token (representación de cada palabra en el vocabulario) a vectores de 'n' dimensiones (word embedding), n: 512.

---

## Características principales
- **Pipeline de datos:** Limpieza, normalización y tokenización del dataset.
- **Inferencia en tiempo real:** Cuando el usuario coloca una oración en inglés, se realiza la traducción a español en tiempo real y se muestra en la interfaz.
- **Interfaz Streamlit:** Permite al usuario interactuar con el modelo de forma directa.

---

## Información adicional importante
- Para cada idioma, creé 1 diccionario que mapea las palabras de ese idioma a un id token, necesario para generar, a partir de él, el word embedding.
- Para que el entrenamiento del modelo sea eficiente y paralelo, utilicé batches. Por ello, debido a que no todas las oraciones tenían el mismo tamaño durante el entrenamiento, utilicé la técnica de 'padding'.

---

## Instalación y uso local

