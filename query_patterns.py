from typing import Dict, List, Any, Optional, Tuple
import re
from typing import List, Tuple, Dict
import configparser
import psycopg2
from fuzzywuzzy import fuzz, process

config_file = "database/database.ini"
env = 'local'

class ValuePatternExtractor:
    def __init__(self, columns_to_check: List[Tuple[str, str]],db_config: Dict[str, str]):
        self.columns_to_check = columns_to_check
        self.db_config = db_config  # New parameter for database connection
        self.cache = {}  # Cache for column values
        self.patterns = self._generate_patterns()

    def get_column_values(self, column_name: str, table_name: str) -> List[str]:
        """Get all unique values for a specific column with caching."""
        cache_key = f"{table_name}.{column_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL")
            values = [str(row[0]) for row in cur.fetchall()]
            cur.close()
            conn.close()
            self.cache[cache_key] = values
            return values
        except Exception as e:
            print(f"Error fetching values for {column_name}: {str(e)}")
            return []

    def find_similar_values(self, value: str, possible_values: List[str], threshold: float = 50) -> List[Tuple[str, int]]:
        """Find similar values using fuzzy matching with fuzzywuzzy."""
        if not value or not possible_values:
            return []

        # Normalize input value and possible values
        value = value.upper()
        possible_values = [str(v).upper() for v in possible_values]

        # Try exact token matching first
        value_tokens = set(value.split())
        exact_matches = []
        for pv in possible_values:
            pv_tokens = set(pv.split())
            if value_tokens.intersection(pv_tokens):
                score = fuzz.token_sort_ratio(value, pv)
                if score >= threshold:
                    exact_matches.append((pv, score))

        if exact_matches:
            return sorted(exact_matches, key=lambda x: x[1], reverse=True)

        # Use multiple fuzzy matching strategies
        matches = []
        for pv in possible_values:
            # Try different fuzzy matching algorithms
            ratio = fuzz.ratio(value, pv)
            partial_ratio = fuzz.partial_ratio(value, pv)
            token_sort = fuzz.token_sort_ratio(value, pv)
            token_set = fuzz.token_set_ratio(value, pv)
            
            # Take the highest score
            best_score = max(ratio, partial_ratio, token_sort, token_set)
            if best_score >= threshold:
                matches.append((pv, best_score))

        return sorted(matches, key=lambda x: x[1], reverse=True)[:5]

    def _generate_patterns(self) -> List[Tuple[str, str, str, str]]:
        # Templates
        EQUALITY_PATTERN = r"(?:{col}|{table}\.{col})\s*=\s*'([^']*)'"
        LIKE_PATTERN = r"(?:{col}|{table}\.{col})\s+LIKE\s+'%([^%]+)%'"
        ILIKE_PATTERN = r"(?:{col}|{table}\.{col})\s+ILIKE\s+'%([^%]+)%'"

        CONTEXT_TEMPLATE_EQ = r"(?i)(?:where|and)\s+(?:\w+\.)?{col}\s*=\s*'[^']*'"
        CONTEXT_TEMPLATE_LIKE = r"(?i)(?:where|and)\s+(?:\w+\.)?{col}\s+LIKE\s+'%[^%]+%'"
        CONTEXT_TEMPLATE_ILIKE = r"(?i)(?:where|and)\s+(?:\w+\.)?{col}\s+ILIKE\s+'%[^%]+%'"

        generated_patterns = []

        for table, column in self.columns_to_check:
            # For equality
            eq_pattern = EQUALITY_PATTERN.format(table=table, col=column)
            eq_context = CONTEXT_TEMPLATE_EQ.format(col=column)
            generated_patterns.append((eq_pattern, table, column, eq_context))

            # For LIKE
            like_pattern = LIKE_PATTERN.format(table=table, col=column)
            like_context = CONTEXT_TEMPLATE_LIKE.format(col=column)
            generated_patterns.append((like_pattern, table, column, like_context))

            # For ILIKE
            ilike_pattern = ILIKE_PATTERN.format(table=table, col=column)
            ilike_context = CONTEXT_TEMPLATE_ILIKE.format(col=column)
            generated_patterns.append((ilike_pattern, table, column, ilike_context))

        return generated_patterns

    def extract_value_patterns(self, query: str) -> List[Tuple[str, str, str, str]]:
        found_patterns = []
        for pattern, table, column, context_pattern in self.patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                context_match = re.search(context_pattern, query, re.IGNORECASE)
                if context_match:
                    context = context_match.group(0)
                    # (extracted_value, table, column, context)
                    found_patterns.append((match.group(1), table, column, context))
        return found_patterns

    def analyze_query_and_suggest(self, failed_query: str) -> Dict[str, List[Tuple[str, int]]]:
        """Analyze failed query and suggest similar values with confidence scores."""
        suggestions = {}
        
        # Extract values and their context from the query
        extracted_patterns = self.extract_value_patterns(failed_query)
        
        for value, table, column, context in extracted_patterns:
            possible_values = self.get_column_values(column, table)
            similar = self.find_similar_values(value, possible_values)
            
            if similar:
                key = f"{table}.{column}"
                suggestions[key] = {
                    'matches': similar,
                    'context': context,
                    'original': value
                }
        
        return suggestions

    def generate_recovery_query(self, failed_query: str, suggestions: Dict[str, Any]) -> str:
        """Generate a new query using suggestions."""
        if not suggestions:
            return failed_query

        recovered_query = failed_query
        for column_key, suggestion_data in suggestions.items():
            if suggestion_data['matches']:
                best_match, score = suggestion_data['matches'][0]
                original = suggestion_data['original']
                context = suggestion_data['context']
                
                # Replace in the specific context rather than globally
                recovered_query = recovered_query.replace(
                    f"'{original}'",
                    f"'{best_match}'"
                )
                
                # If using LIKE, adjust the pattern
                recovered_query = re.sub(
                    rf"LIKE\s+'%{re.escape(original)}%'",
                    f"LIKE '%{best_match}%'",
                    recovered_query,
                    flags=re.IGNORECASE)

        return recovered_query

    def recover_query(self, failed_query: str) -> Tuple[str, Dict[str, Any]]:
        """Main method to recover from query errors."""
        suggestions = self.analyze_query_and_suggest(failed_query)
        if suggestions:
            recovered_query = self.generate_recovery_query(failed_query, suggestions)
            return recovered_query, suggestions
        return failed_query, {}

# Example usage:
if __name__ == "__main__":
    columns_to_check = [
        ("film", "title"),
        ("customer", "first_name"),
        ("customer", "last_name"),
        ("category", "name")
    ]
    config = configparser.ConfigParser()
    config.read(config_file)
    db_config = config[env]

    extractor = ValuePatternExtractor(columns_to_check,db_config)

    # Example query from Pagila:
    # Suppose you have a query that tries to find a film by a slightly misspelled title:
    # Test query
    query = "SELECT * FROM film WHERE title = 'ALIEN CENR' AND category.name LIKE '%Comed%' AND last_name = 'Smith' AND first_name = 'John'"
    query = """SELECT release_year FROM film WHERE title = 'ANEM LUKE';"""
    print(query)    
    # This should find two patterns: one for film.title = 'ACADMY DINOSAURE' and one for category.name LIKE '%Comed%'
    corrected_query, suggestions = extractor.recover_query(query)
    print(corrected_query)
    print("-----")
    print(suggestions)
    # Expected output (something like):
    # [
    #   ("ACADMY DINOSAURE", "film", "title", "WHERE title = 'ACADMY DINOSAURE'"),
    #   ("Comed", "category", "name", "AND category.name LIKE '%Comed%'")
    # ]
