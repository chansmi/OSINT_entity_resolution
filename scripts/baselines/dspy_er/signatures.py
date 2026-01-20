#!/usr/bin/env python3
"""
DSPy Signatures for Entity Resolution.

Defines typed input/output signatures that replace manual prompt engineering.
The docstrings become instructions that optimizers can modify.
"""

from typing import Literal
import dspy


class EntityResolution(dspy.Signature):
    """Determine if two entity records refer to the same real-world entity.

    Primary task: Identify CONFLICTS, not similarities.
    - Name variations (transliterations, nicknames, titles) are common
    - Missing fields are normal - absence of data is NOT evidence of difference
    - Same entity often appears across multiple sources with variations

    Decision Process:
    1. Look for CONTRADICTORY evidence (different dates, conflicting IDs, incompatible attributes)
    2. If NO contradictions found -> POSITIVE (same entity)
    3. Only NEGATIVE if explicit conflicts exist

    The DEFAULT is POSITIVE unless you find proof of difference.
    """

    entity_a: str = dspy.InputField(
        desc="First entity record with names, dates, IDs, and other attributes"
    )
    entity_b: str = dspy.InputField(
        desc="Second entity record with names, dates, IDs, and other attributes"
    )
    classification: Literal["positive", "negative"] = dspy.OutputField(
        desc="positive if same entity, negative if different entities"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of decision, focusing on conflicts or lack thereof"
    )


class EntityResolutionWithConfidence(dspy.Signature):
    """Determine if two entity records refer to the same real-world entity.

    Primary task: Identify CONFLICTS, not similarities.
    - Name variations (transliterations, nicknames, titles) are common
    - Missing fields are normal - absence of data is NOT evidence of difference
    - Same entity often appears across multiple sources with variations

    Decision Process:
    1. Look for CONTRADICTORY evidence (different dates, conflicting IDs, incompatible attributes)
    2. If NO contradictions found -> POSITIVE (same entity)
    3. Only NEGATIVE if explicit conflicts exist

    The DEFAULT is POSITIVE unless you find proof of difference.
    """

    entity_a: str = dspy.InputField(
        desc="First entity record with names, dates, IDs, and other attributes"
    )
    entity_b: str = dspy.InputField(
        desc="Second entity record with names, dates, IDs, and other attributes"
    )
    classification: Literal["positive", "negative"] = dspy.OutputField(
        desc="positive if same entity, negative if different entities"
    )
    confidence: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="Confidence level in the classification"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of decision, focusing on conflicts or lack thereof"
    )


class EntityResolutionTernary(dspy.Signature):
    """Determine if two entity records refer to the same real-world entity.

    Primary task: Identify CONFLICTS, not similarities.
    - Name variations (transliterations, nicknames, titles) are common
    - Missing fields are normal - absence of data is NOT evidence of difference

    Decision Process:
    1. Look for CONTRADICTORY evidence (different dates, conflicting IDs)
    2. If NO contradictions found -> POSITIVE (same entity)
    3. Only NEGATIVE if explicit conflicts exist
    4. UNCERTAIN only when evidence is genuinely ambiguous

    The DEFAULT is POSITIVE unless you find proof of difference.
    """

    entity_a: str = dspy.InputField(
        desc="First entity record with names, dates, IDs, and other attributes"
    )
    entity_b: str = dspy.InputField(
        desc="Second entity record with names, dates, IDs, and other attributes"
    )
    classification: Literal["positive", "negative", "uncertain"] = dspy.OutputField(
        desc="positive if same entity, negative if different, uncertain if genuinely ambiguous"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of decision, focusing on conflicts or lack thereof"
    )


class ConflictDetection(dspy.Signature):
    """Identify specific conflicts between two entity records.

    Look for explicit contradictions in:
    - Birth dates (different years/months/days)
    - National IDs or passport numbers
    - Genders
    - Nationalities (when explicitly different, not just missing)
    - Other identifying attributes

    Report ONLY genuine conflicts, not missing data.
    """

    entity_a: str = dspy.InputField(desc="First entity record")
    entity_b: str = dspy.InputField(desc="Second entity record")
    has_conflicts: bool = dspy.OutputField(
        desc="True if explicit contradictions found, False otherwise"
    )
    conflicts: str = dspy.OutputField(
        desc="List of specific conflicts found, or 'None' if no conflicts"
    )
