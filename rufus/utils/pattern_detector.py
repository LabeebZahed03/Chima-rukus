import re
from collections import defaultdict
from typing import Dict, Set, List, Any, Optional
import logging

class DynamicPatternDetector:
    def __init__(self):
        """Initialize the dynamic pattern detector."""
        self.patterns = defaultdict(lambda: {'count': 0, 'examples': set()})
        self.known_patterns = set()
        self.technical_terms = set()
        self.logger = logging.getLogger(__name__)
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Dynamically analyze text for technical patterns."""
        patterns_found = {
            'syntax_patterns': self._detect_syntax_patterns(text),
            'technical_terms': self._detect_technical_terms(text),
            'structured_data': self._detect_structured_data(text),
            'domain_specific': self._detect_domain_patterns(text)
        }
        
        # Update pattern frequencies
        for category, patterns in patterns_found.items():
            for pattern in patterns:
                self.patterns[pattern]['count'] += 1
                if len(self.patterns[pattern]['examples']) < 5:
                    self.patterns[pattern]['examples'].add(text[:100])
        
        return patterns_found
    
    def _detect_syntax_patterns(self, text: str) -> Set[str]:
        """Detect programming syntax patterns dynamically."""
        patterns = set()
        
        syntax_indicators = [
            (r'\b\w+\s*\([^)]*\)\s*{', 'function_definition'),
            (r'\b(class|interface|enum)\s+\w+', 'type_definition'),
            (r'\b(import|include|require)\b.*?[\n;]', 'import_statement'),
            (r'\b(const|let|var|def|val)\s+\w+\s*=', 'variable_declaration'),
            (r'[</>][\w-]*>', 'markup_tag'),
            (r'[\w-]+\s*:\s*[\w-]+\s*;', 'style_declaration'),
            (r'\b(SELECT|INSERT|UPDATE|DELETE)\b.*?\b(FROM|INTO|WHERE)\b', 'sql_query'),
            (r'@\w+(?:\([^)]*\))?', 'decorator_annotation'),
            (r'\b(async|await|yield|return)\b', 'async_pattern'),
            (r'\b(try|catch|finally|throw)\b', 'error_handling')
        ]
        
        for pattern, pattern_type in syntax_indicators:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                patterns.add(pattern_type)
        
        return patterns
    
    def _detect_technical_terms(self, text: str) -> Set[str]:
        """Detect technical terminology dynamically."""
        terms = set()
        words = text.lower().split()
        
        # Look for compound technical terms
        compound_patterns = [
            r'\b\w+(?:[-_]\w+)+\b',  # hyphenated/underscored terms
            r'\b(?:[A-Z][a-z0-9]+){2,}\b',  # CamelCase
            r'\b[A-Z][A-Z0-9_]+\b'  # CONSTANT_CASE
        ]
        
        for pattern in compound_patterns:
            terms.update(re.findall(pattern, text))
        
        # Detect repeated technical contexts
        window_size = 3
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            if self._is_likely_technical_phrase(window):
                terms.add(' '.join(window))
        
        return terms
    
    def _detect_structured_data(self, text: str) -> Set[str]:
        """Detect structured data patterns."""
        patterns = set()
        
        structure_patterns = [
            (r'[\[\{](?:[^\[\{\}\]]*(?:[\[\{].*?[\}\]])*[^\[\{\}\]]*)*[\}\]]', 'nested_structure'),
            (r'"\w+"\s*:', 'json_like'),
            (r'<\w+>.*?</\w+>', 'xml_like'),
            (r'\b\w+\s*=\s*[\'""].+?[\'""]', 'key_value'),
            (r'\b\d+(?:\.\d+)?(?:px|em|rem|%)\b', 'measurement'),
            (r'\b(?:https?://|www\.)\S+', 'url')
        ]
        
        for pattern, pattern_type in structure_patterns:
            if re.search(pattern, text, re.MULTILINE):
                patterns.add(pattern_type)
        
        return patterns
    
    def _detect_domain_patterns(self, text: str) -> Set[str]:
        """Detect domain-specific patterns dynamically."""
        domains = set()
        
        # Look for domain indicators through context analysis
        context_windows = self._get_context_windows(text)
        for window in context_windows:
            if self._has_technical_context(window):
                domains.update(self._analyze_window_domain(window))
        
        return domains
    
    def _is_likely_technical_phrase(self, words: List[str]) -> bool:
        """Determine if a phrase is likely technical."""
        technical_indicators = {
            'implementation', 'framework', 'library', 'api', 'function',
            'method', 'class', 'object', 'interface', 'protocol',
            'service', 'database', 'query', 'server', 'client'
        }
        
        technical_word_count = sum(1 for word in words if word in technical_indicators)
        has_camelcase = any(re.search(r'[a-z][A-Z]', word) for word in words)
        has_compound = any('-' in word or '_' in word for word in words)
        
        return technical_word_count > 0 or has_camelcase or has_compound
    
    def _get_context_windows(self, text: str, window_size: int = 50) -> List[str]:
        """Get sliding context windows."""
        words = text.split()
        return [
            ' '.join(words[i:i + window_size])
            for i in range(0, len(words), window_size // 2)
        ]
    
    def _has_technical_context(self, text: str) -> bool:
        """Check if text has technical context."""
        technical_terms = self._detect_technical_terms(text)
        syntax_patterns = self._detect_syntax_patterns(text)
        return len(technical_terms) > 0 or len(syntax_patterns) > 0
    
    def _analyze_window_domain(self, text: str) -> Set[str]:
        """Analyze window text for domain identification."""
        domains = set()
        
        # Look for domain-specific keywords and patterns
        if re.search(r'\b(api|rest|http|endpoint)\b', text, re.I):
            domains.add('web_api')
        if re.search(r'\b(database|sql|query|table)\b', text, re.I):
            domains.add('database')
        if re.search(r'\b(html|css|javascript|dom)\b', text, re.I):
            domains.add('web_development')
        if re.search(r'\b(react|component|props|state)\b', text, re.I):
            domains.add('react')
        
        return domains
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns."""
        return {
            'total_patterns': len(self.patterns),
            'frequent_patterns': [
                {
                    'pattern': pattern,
                    'count': data['count'],
                    'examples': list(data['examples'])
                }
                for pattern, data in sorted(
                    self.patterns.items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                )[:10]
            ],
            'technical_terms': list(self.technical_terms)
        }