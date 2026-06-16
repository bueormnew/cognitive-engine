from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from cognitive_engine.core.types import ProcessedInput, SemanticConcept, SemanticState
from cognitive_engine.interfaces.base import SemanticEncoder
from cognitive_engine.utils.numeric import build_numeric_features
from cognitive_engine.utils.text import compress_context, concept_edges, detect_intent, extract_entities, infer_concepts


class TextSemanticEncoder(nn.Module, SemanticEncoder):
    name = "text_semantic_encoder"

    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.to(self.device)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "embedding_dim": self.embedding.embedding_dim}

    def encode(self, processed: ProcessedInput) -> SemanticState:
        tokens = processed.token_tensor
        embedded = self.embedding(tokens)
        transformed = self.transformer(embedded)
        recurrent, _ = self.gru(transformed)
        pooled = self.projection(recurrent.mean(dim=1)).squeeze(0)
        raw_text = str(processed.raw_input)
        intent = processed.metadata.get("intent_hint", detect_intent(raw_text))
        entities = extract_entities(raw_text, processed.tokens)
        inferred = infer_concepts(processed.tokens, entities)
        concepts = [SemanticConcept(label=name, weight=weight, source="hybrid_extractor") for name, weight in inferred]
        graph_edges = concept_edges([concept.label for concept in concepts])
        context = compress_context(intent, [concept.label for concept in concepts], processed.tokens)
        return SemanticState(
            raw_input=processed.raw_input,
            modality=processed.modality,
            sequence_embedding=recurrent.squeeze(0),
            pooled_embedding=pooled,
            intent=intent,
            entities=entities,
            concepts=concepts,
            concept_graph_edges=graph_edges,
            compressed_context=context,
            metadata={"token_count": len(processed.tokens)},
        )


class NumericSemanticEncoder(nn.Module, SemanticEncoder):
    name = "numeric_semantic_encoder"

    def __init__(self, embedding_dim: int = 32, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.op_embedding = nn.Embedding(3, 8)
        self.encoder = nn.Sequential(
            nn.Linear(15, 32),
            nn.Tanh(),
            nn.Linear(32, embedding_dim),
        )
        self.to(self.device)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "embedding_dim": 32}

    def encode(self, processed: ProcessedInput) -> SemanticState:
        op_id = torch.tensor([processed.metadata["operation_id"]], dtype=torch.long, device=self.device)
        op_embed = self.op_embedding(op_id)
        a = processed.metadata["a"]
        b = processed.metadata["b"]
        scale = float(processed.metadata.get("scale", 12.0))
        base_features = torch.tensor([build_numeric_features(a, b, scale)], dtype=torch.float32, device=self.device)
        encoder_input = torch.cat([base_features, op_embed], dim=-1)
        pooled = self.encoder(encoder_input).squeeze(0)
        operation = processed.operation or "add"
        concepts = [
            SemanticConcept(label=operation, weight=1.0, source="numeric_operator"),
            SemanticConcept(label=f"lhs:{a:g}", weight=0.8, source="numeric_operand"),
            SemanticConcept(label=f"rhs:{b:g}", weight=0.8, source="numeric_operand"),
        ]
        return SemanticState(
            raw_input=processed.raw_input,
            modality="numeric",
            sequence_embedding=None,
            pooled_embedding=pooled,
            intent=f"arithmetic_{operation}",
            entities=[],
            concepts=concepts,
            concept_graph_edges=[(concepts[1].label, concepts[0].label, 1.0), (concepts[2].label, concepts[0].label, 1.0)],
            compressed_context=f"Compute {a:g} {operation} {b:g}",
            metadata={
                "numeric_features": base_features.squeeze(0).detach(),
                "operation": operation,
                "operation_id": processed.metadata["operation_id"],
                "target": processed.metadata.get("target"),
                "a": a,
                "b": b,
            },
        )


class HybridSemanticEncoder(SemanticEncoder):
    name = "hybrid_semantic_dispatcher"

    def __init__(self, text_encoder: TextSemanticEncoder, numeric_encoder: NumericSemanticEncoder) -> None:
        self.text_encoder = text_encoder
        self.numeric_encoder = numeric_encoder

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "encoders": [self.text_encoder.name, self.numeric_encoder.name]}

    def encode(self, processed: ProcessedInput) -> SemanticState:
        if processed.modality == "numeric":
            return self.numeric_encoder.encode(processed)
        return self.text_encoder.encode(processed)
