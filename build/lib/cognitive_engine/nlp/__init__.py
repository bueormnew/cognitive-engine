"""
API de Alto Nivel de NLP sobre Cognitive Engine V2.

Permite cargar, guardar y entrenar modelos basados en V2 para tareas específicas
como Traducción o Encoders tipo BERT, con soporte de Hugging Face.
"""

from .models import CognitiveTranslator, CognitiveEncoder, CognitiveModel
from .trainer import CognitiveTrainer

__all__ = [
    "CognitiveModel",
    "CognitiveTranslator",
    "CognitiveEncoder",
    "CognitiveTrainer",
]
