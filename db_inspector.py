from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
import configparser
import os
from typing import Dict, List, Any, Optional


class DVDRentalInspector:
    """
    A class to inspect the dvdrental database and provide comprehensive, structured information.
    The output is designed to be easily consumable by an LLM for SQL generation tasks.
    """
    df_credentials = "database/database.ini"
    def __init__(self, env: str = 'local', config_file: str = df_credentials):
        """
        Initialize the schema inspector with database configuration.
        
        Parameters:
            env (str): The environment section in the configuration file.
            config_file (str): Path to the configuration file containing DB credentials.
        """
        self.config = self._get_db_config(env, config_file)
        db_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}/{self.config['database']}"
        self.engine = create_engine(db_url)
        self.inspector = inspect(self.engine)
        print(db_url)

    def _get_db_config(self, env, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config[env]

    def get_schema_for_prompt(self) -> str:
        """
        Get a concise schema representation focusing on tables, columns, and relationships.
        Ideal for providing minimal context to an LLM for SQL generation.
        """
        lines = []
        tables = self.inspector.get_table_names()
        for table_name in tables:
            columns = [col['name'] for col in self.inspector.get_columns(table_name)]
            fks = self.inspector.get_foreign_keys(table_name)
            line = f"- {table_name} ({', '.join(columns)})"
            if fks:
                for fk in fks:
                    ref_table = fk['referred_table']
                    local_cols = fk['constrained_columns']
                    line += f"\n  Related to {ref_table} via {', '.join(local_cols)}"
            lines.append(line)
        return "\n".join(lines)

if __name__ == "__main__":
    inspector = DVDRentalInspector()
    print(inspector.get_schema_for_prompt())
