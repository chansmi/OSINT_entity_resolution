#!/usr/bin/env python3
"""
DSPy Modules for Entity Resolution.

Provides different module implementations:
- EntityResolver: Basic Predict-based resolver
- ChainOfThoughtResolver: Adds reasoning step before classification
- TwoStageResolver: First detects conflicts, then classifies
"""

import dspy

from .signatures import (
    EntityResolution,
    EntityResolutionWithConfidence,
    EntityResolutionTernary,
    ConflictDetection,
)


class EntityResolver(dspy.Module):
    """Basic entity resolver using dspy.Predict.

    Equivalent to zero-shot classification without explicit reasoning.
    """

    def __init__(self, ternary: bool = False, with_confidence: bool = False):
        """Initialize resolver.

        Args:
            ternary: If True, allow "uncertain" classification
            with_confidence: If True, include confidence in output
        """
        super().__init__()

        if ternary:
            self.predict = dspy.Predict(EntityResolutionTernary)
        elif with_confidence:
            self.predict = dspy.Predict(EntityResolutionWithConfidence)
        else:
            self.predict = dspy.Predict(EntityResolution)

        self.ternary = ternary
        self.with_confidence = with_confidence

    def forward(self, entity_a: str, entity_b: str) -> dspy.Prediction:
        """Classify an entity pair.

        Args:
            entity_a: Formatted string for first entity
            entity_b: Formatted string for second entity

        Returns:
            Prediction with classification and reasoning
        """
        return self.predict(entity_a=entity_a, entity_b=entity_b)


class ChainOfThoughtResolver(dspy.Module):
    """Entity resolver with chain-of-thought reasoning.

    Uses dspy.ChainOfThought to add explicit reasoning before classification.
    This typically improves accuracy by forcing the model to think step-by-step.
    """

    def __init__(self, ternary: bool = False, with_confidence: bool = False):
        """Initialize resolver with CoT.

        Args:
            ternary: If True, allow "uncertain" classification
            with_confidence: If True, include confidence in output
        """
        super().__init__()

        if ternary:
            self.predict = dspy.ChainOfThought(EntityResolutionTernary)
        elif with_confidence:
            self.predict = dspy.ChainOfThought(EntityResolutionWithConfidence)
        else:
            self.predict = dspy.ChainOfThought(EntityResolution)

        self.ternary = ternary
        self.with_confidence = with_confidence

    def forward(self, entity_a: str, entity_b: str) -> dspy.Prediction:
        """Classify an entity pair with chain-of-thought reasoning.

        Args:
            entity_a: Formatted string for first entity
            entity_b: Formatted string for second entity

        Returns:
            Prediction with classification and reasoning
        """
        return self.predict(entity_a=entity_a, entity_b=entity_b)


class TwoStageResolver(dspy.Module):
    """Two-stage entity resolver: conflict detection then classification.

    Stage 1: Explicitly identify any conflicts between entities
    Stage 2: Make final classification based on conflict analysis

    This approach matches the "conflict-focused" prompting strategy
    from the existing baseline but makes it explicit in the pipeline.
    """

    def __init__(self):
        """Initialize two-stage resolver."""
        super().__init__()

        # Stage 1: Detect conflicts
        self.conflict_detector = dspy.ChainOfThought(ConflictDetection)

        # Stage 2: Final classification (uses conflict info)
        self.classifier = dspy.ChainOfThought(EntityResolution)

    def forward(self, entity_a: str, entity_b: str) -> dspy.Prediction:
        """Classify an entity pair using two-stage approach.

        Args:
            entity_a: Formatted string for first entity
            entity_b: Formatted string for second entity

        Returns:
            Prediction with classification, reasoning, and conflict info
        """
        # Stage 1: Detect conflicts
        conflict_result = self.conflict_detector(
            entity_a=entity_a,
            entity_b=entity_b
        )

        # Stage 2: Classify based on conflict analysis
        # Append conflict info to entity descriptions
        entity_a_with_context = f"{entity_a}\n\n[Conflict Analysis: {conflict_result.conflicts}]"
        entity_b_with_context = entity_b

        if conflict_result.has_conflicts:
            # If conflicts found, include them in context
            entity_a_with_context = f"{entity_a}\n\n[CONFLICTS DETECTED: {conflict_result.conflicts}]"

        classification_result = self.classifier(
            entity_a=entity_a_with_context,
            entity_b=entity_b_with_context
        )

        # Combine results
        return dspy.Prediction(
            classification=classification_result.classification,
            reasoning=classification_result.reasoning,
            has_conflicts=conflict_result.has_conflicts,
            conflicts=conflict_result.conflicts,
        )


class ConflictFocusedResolver(dspy.Module):
    """Resolver that explicitly focuses on conflict detection.

    This module is designed to match the existing "conflict-focused"
    prompting strategy while allowing DSPy optimization.
    """

    def __init__(self, with_confidence: bool = False):
        """Initialize conflict-focused resolver.

        Args:
            with_confidence: If True, include confidence in output
        """
        super().__init__()

        if with_confidence:
            self.predict = dspy.ChainOfThought(EntityResolutionWithConfidence)
        else:
            self.predict = dspy.ChainOfThought(EntityResolution)

        self.with_confidence = with_confidence

    def forward(self, entity_a: str, entity_b: str) -> dspy.Prediction:
        """Classify using conflict-focused reasoning.

        Args:
            entity_a: Formatted string for first entity
            entity_b: Formatted string for second entity

        Returns:
            Prediction with classification and reasoning
        """
        # Add conflict-focused instruction to the entity context
        instruction = (
            "Focus on finding CONFLICTS between these entities. "
            "If no explicit contradictions are found, classify as POSITIVE."
        )
        entity_a_augmented = f"{instruction}\n\n{entity_a}"

        return self.predict(entity_a=entity_a_augmented, entity_b=entity_b)


# --- Module Registry ---

MODULES = {
    "basic": EntityResolver,
    "cot": ChainOfThoughtResolver,
    "two_stage": TwoStageResolver,
    "conflict_focused": ConflictFocusedResolver,
}


def get_module(name: str, **kwargs) -> dspy.Module:
    """Get a resolver module by name.

    Args:
        name: Module name ("basic", "cot", "two_stage", "conflict_focused")
        **kwargs: Arguments to pass to the module constructor

    Returns:
        Initialized DSPy module

    Raises:
        ValueError: If module name is unknown
    """
    if name not in MODULES:
        raise ValueError(f"Unknown module: {name}. Available: {list(MODULES.keys())}")

    return MODULES[name](**kwargs)
