# Cognitive Engine NLP Library

Una librería de Python de alto nivel para construir, entrenar y desplegar modelos de Inteligencia Artificial basados en la Arquitectura Cognitiva Modular V2.

Esta librería facilita la adopción de las ventajas de la Arquitectura V2 (Aprendizaje Continuo Selectivo, Memoria Jerárquica, Enrutamiento Dinámico) en tareas clásicas de NLP como:
- Modelos Text-to-Text (e.g. Traducción, Resumen).
- Encoders para representación densa (e.g. tipo BERT).

## Instalación

Puedes instalar la librería y sus dependencias (incluyendo soporte para Hugging Face) directamente desde el código fuente o mediante `pip` si está publicada:

```bash
pip install cognitive-engine
# o desde el código fuente:
pip install .
```

## Dependencias

- `torch>=2.12`
- `transformers>=4.30.0`
- `datasets>=2.14.0`
- Y dependencias internas como `networkx`, `numpy`, etc.

## Estructura de la API de Alto Nivel

La API se encuentra en `cognitive_engine.nlp` y expone un flujo de trabajo muy similar a las librerías tradicionales de Deep Learning, pero operando sobre la Memoria y el Routing de V2.

- **`CognitiveTranslator`**: Modelo pre-configurado para tareas generativas (traducción).
- **`CognitiveEncoder`**: Modelo pre-configurado para extraer representaciones contextuales.
- **`CognitiveTrainer`**: Motor de entrenamiento que mapea `epochs` a ciclos de Inyección de Memoria y Consolidación Plástica.

## Ejemplo de Uso: Traducción Inglés a Español

```python
from cognitive_engine.nlp import CognitiveTranslator, CognitiveTrainer
from datasets import load_dataset

# 1. Cargar Dataset (ejemplo desde HuggingFace)
dataset = load_dataset("opus_books", "en-es", split="train[:100]")

# 2. Inicializar Modelo de Traducción
modelo = CognitiveTranslator(source_lang="en", target_lang="es")

# 3. Entrenar el Modelo (Aprendizaje Continuo V2)
trainer = CognitiveTrainer(
    model=modelo,
    train_dataset=dataset["translation"],
    args={"num_train_epochs": 3, "consolidation_steps": 50}
)
trainer.train()

# 4. Inferir
respuesta = modelo.translate("The architecture is very robust.", allow_learning=False)
print("Traducción:", respuesta)

# 5. Guardar el Modelo (incluye su Memoria)
modelo.save_pretrained("./mi_modelo_traductor")

# 6. Cargar el Modelo desde el disco
modelo_cargado = CognitiveTranslator.from_pretrained("./mi_modelo_traductor")
```

## Arquitectura Interna

La librería envuelve:
- **`SemanticBackboneV2`**: Como encoder principal.
- **`StableCoreV2`**: Como generador o reasoner.
- **`MemorySystem`**: Donde se guarda el conocimiento aprendido de los datasets.
- **`ConsolidationEngine`**: Ejecutado periódicamente por el `CognitiveTrainer` para fusionar patrones y abstraer conocimiento.

Para un análisis detallado del engine V2, revisa los documentos en `docs/`.
