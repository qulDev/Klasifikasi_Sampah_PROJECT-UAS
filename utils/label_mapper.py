"""
Label mapping and standardization for waste classification.

Provides fuzzy matching, keyword extraction, and fallback strategies
to map heterogeneous class names to standardized target classes.
"""

from typing import Dict, List, Tuple

from rapidfuzz import fuzz, process


# Target classes for waste classification
TARGET_CLASSES = ["plastic", "metal", "glass", "paper", "cardboard", "other"]

# Material keywords for intelligent mapping
MATERIAL_KEYWORDS = {
    "plastic": ["plastic", "pet", "hdpe", "pvc", "ldpe", "pp", "ps", "bottle", "wrapper", "bag"],
    "metal": ["metal", "aluminum", "steel", "can", "tin", "foil", "aluminium"],
    "glass": ["glass", "jar", "bottle"],
    "paper": ["paper", "newspaper", "magazine", "book", "cardboard", "carton"],
    "cardboard": ["cardboard", "carton", "box", "packaging"],
}

# Manual class mappings for specific datasets
# Maps source class names to target classes
MANUAL_CLASS_MAPPINGS = {
    # garbage-classification-v2 additional classes -> other
    "battery": "other",
    "biological": "other", 
    "clothes": "other",
    "shoes": "other",
    "trash": "other",
    # Standard mappings (explicit)
    "plastic": "plastic",
    "metal": "metal",
    "glass": "glass",
    "paper": "paper",
    "cardboard": "cardboard",
}


def normalize_label(label: str) -> str:
    """
    Normalize a label string for comparison.

    Args:
        label: Raw label string

    Returns:
        Lowercase, stripped, underscore-replaced label

    Example:
        >>> normalize_label("Plastic_Bottle")
        'plastic bottle'
    """
    return label.lower().strip().replace('_', ' ').replace('-', ' ')


def map_label_exact(source_label: str, target_classes: List[str] = None) -> Tuple[str, str, float]:
    """
    Attempt exact case-insensitive match.

    Args:
        source_label: Label to map
        target_classes: List of target class names (default: TARGET_CLASSES)

    Returns:
        Tuple of (target_class, method, confidence)
        - target_class: Mapped class name or None if no match
        - method: Mapping method used
        - confidence: Confidence score [0, 1]

    Example:
        >>> target, method, conf = map_label_exact("PLASTIC")
        >>> print(f"{target} via {method} ({conf})")
        plastic via exact (1.0)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    normalized = normalize_label(source_label)
    
    for target in target_classes:
        if normalized == target:
            return target, "exact", 1.0
    
    return None, "exact", 0.0


def map_label_fuzzy(source_label: str, target_classes: List[str] = None, threshold: int = 80) -> Tuple[str, str, float]:
    """
    Attempt fuzzy string matching using Levenshtein distance.

    Args:
        source_label: Label to map
        target_classes: List of target class names (default: TARGET_CLASSES)
        threshold: Minimum similarity score (0-100)

    Returns:
        Tuple of (target_class, method, confidence)

    Example:
        >>> target, method, conf = map_label_fuzzy("platic")  # typo
        >>> print(f"{target} via {method} ({conf:.2f})")
        plastic via fuzzy (0.83)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    normalized = normalize_label(source_label)
    
    # Find best fuzzy match
    result = process.extractOne(
        normalized,
        target_classes,
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )
    
    if result:
        target, score, _ = result
        return target, "fuzzy", score / 100.0
    
    return None, "fuzzy", 0.0


def map_label_substring(source_label: str, target_classes: List[str] = None) -> Tuple[str, str, float]:
    """
    Check if any target class is a substring of the source label.

    Args:
        source_label: Label to map
        target_classes: List of target class names (default: TARGET_CLASSES)

    Returns:
        Tuple of (target_class, method, confidence)

    Example:
        >>> target, method, conf = map_label_substring("plastic_bottle_clear")
        >>> print(f"{target} via {method} ({conf})")
        plastic via substring (0.95)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    normalized = normalize_label(source_label)
    
    for target in target_classes:
        if target in normalized:
            return target, "substring", 0.95
    
    return None, "substring", 0.0


def map_label_keyword(source_label: str) -> Tuple[str, str, float]:
    """
    Map using material keywords (e.g., "aluminum_can" -> "metal").

    Args:
        source_label: Label to map

    Returns:
        Tuple of (target_class, method, confidence)

    Example:
        >>> target, method, conf = map_label_keyword("aluminum_foil")
        >>> print(f"{target} via {method} ({conf})")
        metal via keyword (0.90)
    """
    normalized = normalize_label(source_label)
    
    for target_class, keywords in MATERIAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                return target_class, "keyword", 0.90
    
    return None, "keyword", 0.0


def map_label(
    source_label: str,
    target_classes: List[str] = None,
    manual_mappings: Dict[str, str] = None
) -> Tuple[str, str, float]:
    """
    Map a source label to a target class using multiple strategies.

    Strategies applied in order:
    1. Manual override (if provided)
    2. Exact match (case-insensitive)
    3. Fuzzy match (Levenshtein distance)
    4. Substring match
    5. Keyword match
    6. Fallback to "other"

    Args:
        source_label: Label to map
        target_classes: List of target class names (default: TARGET_CLASSES)
        manual_mappings: Optional dict of source -> target overrides

    Returns:
        Tuple of (target_class, method, confidence)

    Example:
        >>> target, method, conf = map_label("PET_bottle")
        >>> print(f"Mapped '{target}' via {method} ({conf:.2f})")
        Mapped 'plastic' via keyword (0.90)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    normalized = normalize_label(source_label)

    # 1. Manual override - check both provided mappings and default MANUAL_CLASS_MAPPINGS
    if manual_mappings and normalized in manual_mappings:
        return manual_mappings[normalized], "manual", 1.0
    
    # Check default manual mappings
    if normalized in MANUAL_CLASS_MAPPINGS:
        return MANUAL_CLASS_MAPPINGS[normalized], "manual", 1.0

    # 2. Exact match
    target, method, conf = map_label_exact(source_label, target_classes)
    if target:
        return target, method, conf

    # 3. Fuzzy match
    target, method, conf = map_label_fuzzy(source_label, target_classes)
    if target:
        return target, method, conf

    # 4. Substring match
    target, method, conf = map_label_substring(source_label, target_classes)
    if target:
        return target, method, conf

    # 5. Keyword match
    target, method, conf = map_label_keyword(source_label)
    if target:
        return target, method, conf

    # 6. Fallback to "other"
    return "other", "fallback", 0.50
