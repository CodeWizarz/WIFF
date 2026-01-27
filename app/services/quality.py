import re
from typing import Dict, Any

class QualityFilter:
    """
    Quality control for ingestion content.
    Filters out low-value, spammy, or sensitive content.
    """
    
    def __init__(self):
        # Regex patterns for PII
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
        
        # Boilerplate indicators
        self.boilerplate_terms = [
            "all rights reserved", "copyright", "terms of service", 
            "privacy policy", "unsubscribe"
        ]

    def is_worth_storing(self, text: str) -> bool:
        """
        Check if text meets minimum quality standards.
        """
        words = text.split()
        
        # 1. Length check
        if len(words) < 10:
            return False
            
        # 2. Boilerplate check (simple heuristic)
        text_lower = text.lower()
        if any(term in text_lower for term in self.boilerplate_terms):
            # If boilerplate dominates the text, skip it
            # (Heuristic: if text is short and contains boilerplate)
            if len(words) < 50:
                return False
                
        # 3. Repetition check (spam detection)
        # Calculate unique word ratio
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Very repetitious
                return False
                
        return True

    def contains_pii(self, text: str) -> bool:
        """
        Check for Personally Identifiable Information.
        """
        for pattern in self.pii_patterns.values():
            if re.search(pattern, text):
                return True
        return False
