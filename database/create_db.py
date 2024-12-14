import requests
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import configparser
import sys

def get_config(env='local'):
    """Read database connection parameters."""

    config = configparser.ConfigParser()
    config.read('database.ini')
    
    if env not in config:
        raise Exception(f'Environment {env} not found in database.ini')
    
    return config[env]

def execute_sql_file(cursor, sql_content, user):
    """Execute SQL statements from a file content."""
    current_statement = []
    in_function = False
    in_copy = False
    dollar_tag = None
    copy_data = []

    for line in sql_content.split('\n'):
        line = line.rstrip()  # Keep trailing tabs for COPY data
        
        # Skip comments and empty lines
        if not line or line.startswith('--'):
            continue

        # Handle COPY statements
        if line.startswith('COPY ') and ' FROM stdin;' in line:
            # Extract table name and columns
            table_start = line.find('public.') + 7
            table_end = line.find(' (', table_start)
            if table_end == -1:
                table_end = line.find(' FROM stdin;', table_start)
            table_name = line[table_start:table_end]
            
            # Extract columns if present
            columns = None
            if '(' in line and ')' in line:
                cols_start = line.find('(') + 1
                cols_end = line.find(')')
                columns = [c.strip() for c in line[cols_start:cols_end].split(',')]
            
            in_copy = True
            current_statement = []
            continue
        
        if in_copy:
            if line == r'\.':  # Use raw string for the dot escape sequence
                # End of COPY data
                in_copy = False
                if copy_data:
                    # Create StringIO object for copy_from
                    from io import StringIO
                    copy_buffer = StringIO('\n'.join(copy_data) + '\n')
                    try:
                        cursor.copy_expert(f"COPY public.{table_name} FROM STDIN", copy_buffer)
                    except Exception as e:
                        print(f"Error copying data into {table_name}: {str(e)}")
                        raise
                    copy_data = []
            else:
                copy_data.append(line)
            continue

        # Replace postgres user with current user
        if 'TO postgres;' in line:
            line = line.replace('TO postgres;', f'TO {user};')

        # Check for dollar-quoted strings
        if not in_function and 'AS $_$' in line:
            in_function = True
            dollar_tag = '$_$'
        elif not in_function and 'AS $$' in line:
            in_function = True
            dollar_tag = '$$'

        current_statement.append(line)

        # Check if we've reached the end of a function definition
        if in_function and dollar_tag in line and line.endswith(';'):
            in_function = False
            # Execute the complete function definition
            sql = ' '.join(current_statement)
            try:
                cursor.execute(sql)
            except Exception as e:
                if 'OWNER TO' not in sql:
                    print(f"Error executing statement: {str(e)}")
                    print(f"Failed statement: {sql[:200]}...")
                    raise
            current_statement = []
            continue

        # For non-function statements, split on semicolon
        if not in_function and line.endswith(';'):
            sql = ' '.join(current_statement)
            try:
                cursor.execute(sql)
            except Exception as e:
                if 'OWNER TO' not in sql:
                    print(f"Error executing statement: {str(e)}")
                    print(f"Failed statement: {sql[:200]}...")
                    raise
            current_statement = []

def create_database(config):
    """Create and populate the PostgreSQL database with Sakila data."""
    try:
        # First, connect to PostgreSQL server to create the database
        conn = psycopg2.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database='postgres'  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        database_name = config['database']
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{database_name}'")
        exists = cursor.fetchone()
        if exists:
            # Close all connections to the database before dropping
            cursor.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{database_name}'
                AND pid <> pg_backend_pid()
            """)
            cursor.execute(f'DROP DATABASE IF EXISTS {database_name}')
            print(f"Dropped existing database: {database_name}")
        
        cursor.execute(f'CREATE DATABASE {database_name}')
        print(f"Created database: {database_name}")
        
        cursor.close()
        conn.close()

        # Connect to the new database
        conn = psycopg2.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=database_name
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Download and execute schema
        print("\nDownloading and executing schema...")
        schema_url = "https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-schema.sql"
        schema_response = requests.get(schema_url)
        if schema_response.status_code != 200:
            raise Exception(f"Failed to download schema. Status code: {schema_response.status_code}")
        
        schema_content = schema_response.text
        execute_sql_file(cursor, schema_content, config['user'])
        print("Schema created successfully")
        
        # Download and execute data
        print("\nDownloading and executing data...")
        data_url = "https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-data.sql"
        data_response = requests.get(data_url)
        if data_response.status_code != 200:
            raise Exception(f"Failed to download data. Status code: {data_response.status_code}")
        
        data_content = data_response.text
        execute_sql_file(cursor, data_content, config['user'])
        print("Data imported successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

def print_database_stats(config):
    """Print statistics about the database."""
    try:
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        
        tables = ['actor', 'category', 'film', 'film_actor', 'film_category', 
                'language', 'country', 'city', 'address', 'store', 'staff', 
                'customer', 'inventory', 'rental', 'payment']
        
        print("\nDatabase Statistics:")
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"- {table.capitalize()}: {count:,} records")
            except Exception as e:
                print(f"Error getting count for {table}: {str(e)}")
                conn.rollback()
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python create_postgres_rental_db.py <environment>")
        print("Example: python create_postgres_rental_db.py local")
        sys.exit(1)
    
    env = sys.argv[1]
    config = get_config(env)
    
    if create_database(config):
        print_database_stats(config)
    
    print("\nDatabase creation completed successfully!")