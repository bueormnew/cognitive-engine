from cognitive_engine.nlp import CognitiveTranslator, CognitiveTrainer

def main():
    print("Demostración de API de Alto Nivel NLP - Traducción")
    
    # Simulación de carga de dataset con Hugging Face `datasets`
    # (Para no hacer descargas pesadas en la demo real, usamos un dummy similar)
    print("1. Cargando dataset dummy de Inglés-Español...")
    dummy_dataset = [
        {"en": "Hello world", "es": "Hola mundo"},
        {"en": "The cognitive architecture is modular.", "es": "La arquitectura cognitiva es modular."},
        {"en": "Continuous learning prevents catastrophic forgetting.", "es": "El aprendizaje continuo previene el olvido catastrófico."},
        {"en": "Dependency injection is useful.", "es": "La inyección de dependencias es útil."},
        {"en": "We use a stable core and a plastic learner.", "es": "Usamos un núcleo estable y un aprendiz plástico."}
    ]
    
    print("2. Inicializando CognitiveTranslator...")
    translator = CognitiveTranslator(source_lang="en", target_lang="es")
    
    print("3. Preparando CognitiveTrainer...")
    trainer = CognitiveTrainer(
        model=translator,
        train_dataset=dummy_dataset,
        args={"num_train_epochs": 2, "consolidation_steps": 3}
    )
    
    print("4. Entrenando el modelo...")
    trainer.train()
    
    print("\\n5. Evaluando el modelo (Inferencia V2)...")
    test_sentence = "The architecture is modular."
    print(f"Input: {test_sentence}")
    response = translator.translate(test_sentence, allow_learning=False)
    print(f"Output: {response}")
    
    print("\\n6. Guardando el modelo entrenado a './artifacts/translator_model'...")
    translator.save_pretrained("./artifacts/translator_model")
    
    print("7. Recargando el modelo...")
    loaded_model = CognitiveTranslator.from_pretrained("./artifacts/translator_model")
    print("Modelo recargado exitosamente con su estado V2.")

if __name__ == "__main__":
    main()
