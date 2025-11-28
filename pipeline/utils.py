"""
Utility functions.
"""
import re
from typing import Set


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "trainable_percentage": round(100 * trainable_params / total_params, 2)
    }


def extract_components(text: str) -> Set[str]:
    """
    Extract architectural components using regex.
    """
    components = set()
    
    # Pattern 1: File names
    file_pattern = r'\b\w+\.(java|py|ts|tsx|jsx|js|xml|yaml|yml|json|properties|conf)\b'
    files = re.findall(file_pattern, text, re.IGNORECASE)
    components.update(files)
    
    # Pattern 2: Class/Service names
    class_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:Service|Controller|Repository|Model|Manager|Handler|Provider|Factory|Builder|Adapter|Facade|Strategy)\b'
    classes = re.findall(class_pattern, text)
    components.update(classes)
    
    # Pattern 3: Package/Module names
    package_pattern = r'\b[a-z]+(?:\.[a-z]+){2,}\b'
    packages = re.findall(package_pattern, text)
    components.update(packages)
    
    # Pattern 4: Architecture layer keywords
    layer_keywords = r'\b(controller|service|repository|model|view|database|api|frontend|backend|middleware|gateway|proxy)\b'
    layers = re.findall(layer_keywords, text, re.IGNORECASE)
    components.update([l.lower() for l in layers])
    
    return components