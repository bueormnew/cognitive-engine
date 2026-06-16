import time
from typing import Optional, Dict, Any, List


class CognitiveTrainer:
    """
    Loop de entrenamiento continuo para un modelo Cognitive Engine V2.
    
    Detecta automáticamente la estructura del dataset para soportar múltiples formatos:
    
    - **Pares anidados** (opus_books, wmt, etc.):
      ``{'translation': {'en': '...', 'es': '...'}}``
    
    - **Pares planos con códigos de idioma**:
      ``{'en': '...', 'es': '...'}``
    
    - **Campos genéricos**:
      ``{'source': '...', 'target': '...'}``
      ``{'input': '...', 'output': '...'}``
      ``{'text': '...', 'label': '...'}``
    
    - **Listas de texto plano** (solo fuente, sin target):
      ``{'text': '...'}``

    - **Pares explícitos** pasados como ``(source, target)`` tuples.
    """

    # Campos genéricos de pares que se intentarán en orden
    _PAIR_FIELD_CANDIDATES = [
        ("source", "target"),
        ("input", "output"),
        ("question", "answer"),
        ("text", "label"),
        ("sentence1", "sentence2"),
        ("premise", "hypothesis"),
    ]

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        args: Dict[str, Any] = None,
        source_lang: str = None,
        target_lang: str = None,
        source_field: str = None,
        target_field: str = None,
    ):
        """
        Args:
            model: El modelo CognitiveModel a entrenar.
            train_dataset: Dataset de entrenamiento. Puede ser un HuggingFace Dataset,
                           una lista de dicts, o una lista de tuplas (source, target).
            eval_dataset: Dataset de evaluación opcional.
            args: Configuración del training:
                - ``num_train_epochs`` (int, default 1)
                - ``consolidation_steps`` (int, default 500)
                - ``log_steps`` (int, default 10)
            source_lang: Código de idioma fuente (ej. ``"en"``). Si se especifica,
                         buscará directamente este campo o dentro de ``translation``.
            target_lang: Código de idioma objetivo (ej. ``"es"``).
            source_field: Nombre del campo fuente para override manual.
            target_field: Nombre del campo objetivo para override manual.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or {}
        self.source_lang = source_lang or getattr(model, "source_lang", None)
        self.target_lang = target_lang or getattr(model, "target_lang", None)
        self.source_field = source_field
        self.target_field = target_field

    # ------------------------------------------------------------------
    # Detección automática de estructura
    # ------------------------------------------------------------------

    def _extract_pair(self, example: Any):
        """
        Extrae el par (source_text, target_text) de cualquier formato de ejemplo.
        Retorna (source_text, target_text) donde target_text puede ser None.
        """
        # Caso 1: la entrada es una tupla/lista directa (source, target)
        if isinstance(example, (tuple, list)) and len(example) >= 2:
            return str(example[0]), str(example[1])

        # Caso 2: la entrada es un string plano
        if isinstance(example, str):
            return example, None

        if not isinstance(example, dict):
            return str(example), None

        # Caso 3: override manual de campos
        if self.source_field and self.target_field:
            src = example.get(self.source_field, "")
            tgt = example.get(self.target_field, "")
            return str(src), str(tgt) if tgt else None

        # Caso 4: estructura anidada tipo opus_books/wmt
        # {'translation': {'en': '...', 'es': '...'}}
        if "translation" in example and isinstance(example["translation"], dict):
            trans = example["translation"]
            src_lang = self.source_lang
            tgt_lang = self.target_lang
            # Si no se especificaron idiomas, inferir del primer par de claves
            if not src_lang or not tgt_lang:
                keys = list(trans.keys())
                src_lang = keys[0] if len(keys) > 0 else None
                tgt_lang = keys[1] if len(keys) > 1 else None
            src = trans.get(src_lang, "")
            tgt = trans.get(tgt_lang, "")
            return str(src), str(tgt) if tgt else None

        # Caso 5: pares planos con códigos de idioma {'en': '...', 'es': '...'}
        if self.source_lang and self.target_lang:
            if self.source_lang in example and self.target_lang in example:
                return str(example[self.source_lang]), str(example[self.target_lang])

        # Caso 6: campos genéricos conocidos
        for src_field, tgt_field in self._PAIR_FIELD_CANDIDATES:
            if src_field in example:
                src = example[src_field]
                tgt = example.get(tgt_field)
                return str(src), str(tgt) if tgt else None

        # Caso 7: cualquier campo de texto disponible
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 0:
                return value, None

        return str(example), None

    def _build_prompt(self, source: str, target: str | None) -> str:
        """Construye el prompt de aprendizaje con marcadores que activan el intent correcto."""
        if target:
            if self.source_lang and self.target_lang:
                return f"Learn this translation from {self.source_lang} to {self.target_lang}: '{source}' -> '{target}'"
            return f"Learn this: '{source}' -> '{target}'"
        return f"Learn this: {source}"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Inicia el proceso de aprendizaje continuo y consolidación."""
        epochs = self.args.get("num_train_epochs", 1)
        consolidation_steps = self.args.get("consolidation_steps", 500)
        log_steps = self.args.get("log_steps", 10)

        print(f"Iniciando Cognitive Training por {epochs} épocas...")
        print(f"  Dataset size: {len(self.train_dataset)} ejemplos")

        global_step = 0
        learned_count = 0

        for epoch in range(epochs):
            print(f"\n--- Época {epoch + 1}/{epochs} ---")

            for i, example in enumerate(self.train_dataset):
                source_text, target_text = self._extract_pair(example)

                if not source_text.strip():
                    continue

                training_prompt = self._build_prompt(source_text, target_text)

                response = self.model.engine.process(
                    training_prompt,
                    allow_learning=True,
                )

                global_step += 1
                if response.learning_applied:
                    learned_count += 1

                if global_step % log_steps == 0:
                    learned_rate = learned_count / global_step * 100
                    learning_status = "✅" if response.learning_applied else "❌"
                    print(
                        f"Step {global_step:>6} | "
                        f"Learned: {learning_status} | "
                        f"Learn rate: {learned_rate:.1f}% | "
                        f"Traces: {len(response.traces)}"
                    )

                if global_step % consolidation_steps == 0:
                    print("  --> Ejecutando Consolidación Cognitiva (Deep Sleep)...")
                    report = self.model.engine.consolidator.run()
                    print(f"  --> Consolidación completada. Registros fusionados: {report.merged_records}")

        print(f"\nEntrenamiento completado.")
        print(f"  Total steps: {global_step} | Learned: {learned_count} ({learned_count/max(global_step,1)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        """Evalúa el modelo sin aplicar aprendizaje."""
        if not self.eval_dataset:
            print("No se proporcionó dataset de validación.")
            return

        print("\nIniciando Evaluación Cognitiva...")
        total = len(self.eval_dataset)

        for i, example in enumerate(self.eval_dataset):
            source_text, target_text = self._extract_pair(example)

            if hasattr(self.model, "translate"):
                prediction = self.model.translate(source_text, allow_learning=False)
            else:
                prediction = self.model.engine.process(source_text, allow_learning=False).text

            if i < 3:  # Mostrar algunos ejemplos
                print(f"  [{i+1}] Input:    {source_text[:80]}")
                print(f"       Expected: {(target_text or 'N/A')[:80]}")
                print(f"       Got:      {prediction[:80]}")

        print("Evaluación completada.")
        return {"eval_loss": 0.0, "notes": "Qualitative evaluation done."}

