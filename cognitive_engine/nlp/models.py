import os
import json
import torch
from pathlib import Path
from cognitive_engine import EngineBuilder
from cognitive_engine.core.engine import CognitiveEngine

class CognitiveModel:
    """
    Clase base para todos los modelos NLP construidos sobre Cognitive Engine V2.
    Provee funcionalidades para guardar, cargar y utilizar la arquitectura subyacente.
    """
    def __init__(self, engine: CognitiveEngine, config_dict: dict = None):
        self.engine = engine
        self.config_dict = config_dict or {}

    @classmethod
    def from_pretrained(cls, path: str):
        """
        Carga un modelo previamente guardado.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"La ruta {path} no existe.")

        # Cargar configuración si existe
        config_path = path / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        # En un escenario real, cargaríamos el state_dict de los submodelos (StableCore, etc)
        # Aquí reconstruimos el engine con la configuración.
        engine = EngineBuilder(config_path="configs/default.yaml").build_v2()

        # Simulamos la carga de pesos si existen
        weights_path = path / "pytorch_model.bin"
        if weights_path.exists():
            # engine.stable_core.load_state_dict(torch.load(weights_path))
            pass

        return cls(engine=engine, config_dict=config_dict)

    def save_pretrained(self, path: str):
        """
        Guarda el modelo, configuración y estado de la memoria en disco.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Guardar config
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config_dict, f, indent=2)

        # Guardar snapshot del engine (memoria, replay, etc)
        snapshot = self.engine.snapshot()
        with open(path / "cognitive_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        # Guardar pesos (Simulado)
        # torch.save(self.engine.stable_core.state_dict(), path / "pytorch_model.bin")
        # Por ahora creamos un dummy
        with open(path / "pytorch_model.bin", "w") as f:
            f.write("dummy weights")

        print(f"Modelo V2 guardado exitosamente en {path}")


class CognitiveTranslator(CognitiveModel):
    """
    Modelo optimizado para tareas Text-to-Text (e.g. Traducción).
    Usa el Stable Core para generar texto a partir del contexto recuperado en memoria.
    """
    def __init__(self, engine=None, source_lang="en", target_lang="es"):
        if engine is None:
            engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
        config_dict = {"source_lang": source_lang, "target_lang": target_lang, "task": "translation"}
        super().__init__(engine, config_dict)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text: str, allow_learning: bool = False) -> str:
        """
        Traduce el texto dado, utilizando el pipeline cognitivo V2 completo.
        """
        prompt = f"Translate from {self.source_lang} to {self.target_lang}: {text}"
        # Procesamos usando el engine. Si es inferencia pura, no aprendemos en linea.
        response = self.engine.process(prompt, allow_learning=allow_learning)
        return response.text

    def __call__(self, text: str) -> str:
        return self.translate(text)


class CognitiveEncoder(CognitiveModel):
    """
    Modelo optimizado para extraer representaciones densas (embeddings) tipo BERT.
    """
    def __init__(self, engine=None):
        if engine is None:
            engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
        config_dict = {"task": "encoding"}
        super().__init__(engine, config_dict)

    def encode(self, text: str):
        """
        Devuelve el vector de embedding contextual de la entrada usando el Semantic Backbone.
        """
        # Simulamos que tenemos acceso al payload text
        from cognitive_engine.core.types import TextPayload
        payload = TextPayload(text=text)
        
        # Obtenemos el estado semántico directamente del encoder del engine
        semantic_state = self.engine.semantic_encoder.encode(payload)
        return semantic_state.pooled_embedding
