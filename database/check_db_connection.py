import configparser
import os
from sqlalchemy import create_engine, inspect, text

def get_config(env='local'):
    """Read database credentials from database.ini."""
    if not os.path.exists('database.ini'):
        raise FileNotFoundError("No database.ini file found. Please provide one with your database credentials.")
    
    config = configparser.ConfigParser()
    config.read('database.ini')

    if env not in config:
        raise ValueError(f"Environment '{env}' not found in database.ini.")
    
    return config[env]

def get_engine(db_config):
    """Create and return a SQLAlchemy engine based on the provided configuration."""
    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    return create_engine(db_url)

def get_tables(engine):
    """Return a list of table names in the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()

def get_columns(engine, table_name):
    """Return a list of column names for the given table."""
    inspector = inspect(engine)
    columns_info = inspector.get_columns(table_name)
    return [col['name'] for col in columns_info]

def get_table_row_count(engine, table_name):
    """Return the number of rows in the given table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()

def main():
    # Load the database configuration
    db_config = get_config('local')
    engine = get_engine(db_config)

    # Get the tables
    tables = get_tables(engine)
    print("Tables in the database:")
    for t in tables:
        print(f" - {t}")

    # Get columns for each table
    print("\nTable columns:")
    for t in tables:
        cols = get_columns(engine, t)
        print(f"{t}: {', '.join(cols)}")

    # Get row counts for each table
    print("\nRow counts:")
    for t in tables:
        count = get_table_row_count(engine, t)
        print(f"{t}: {count} rows")

if __name__ == "__main__":
    main()
