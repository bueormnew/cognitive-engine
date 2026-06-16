from cognitive_engine import EngineBuilder
from cognitive_engine.config.schema import EngineConfig, NumericDemoConfig
from cognitive_engine.utils.numeric import enumerate_numeric_dataset


def test_plastic_module_learns_small_numeric_domain():
    config = EngineConfig(numeric_demo=NumericDemoConfig(max_operand=5, epochs=10, replay_batch_size=16))
    builder = EngineBuilder(config=config)
    engine = builder.build()
    dataset = enumerate_numeric_dataset(max_operand=5, operations=["add", "sub", "mul"], seed=7, scale=5)
    split = int(len(dataset) * 0.75)
    train_data = dataset[:split]
    val_data = dataset[split:]

    for _ in range(10):
        for sample in train_data:
            engine.process(
                {
                    "modality": "numeric",
                    "a": sample["a"],
                    "b": sample["b"],
                    "operation": sample["operation"],
                    "target": sample["target"],
                },
                allow_learning=True,
            )

    metrics = engine.trainer.evaluate(val_data)
    assert metrics["accuracy"] >= 0.8
    assert metrics["mae"] < 1.5
