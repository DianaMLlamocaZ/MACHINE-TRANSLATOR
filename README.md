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
- **Embedding:** Convierte el input token (representación de cada palabra en el vocabulario) a vectores de 'n' dimensiones, n: 512.

