from typing import Annotated, TypedDict, Literal, Optional, List, Dict, Any
from operator import add
from datetime import datetime
from dotenv import load_dotenv
import json
import configparser
import psycopg2
from psycopg2.extras import RealDictCursor
from decimal import Decimal
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from db_inspector import DVDRentalInspector
from query_patterns import ValuePatternExtractor

load_dotenv()

config_file = "database/database.ini"
env = 'local'

config = configparser.ConfigParser()
config.read(config_file)
db_config = config[env]

columns_to_check = [
    ("film", "title"),
    ("customer", "first_name"),
    ("customer", "last_name"),
    ("category", "name")
]

extractor = ValuePatternExtractor(columns_to_check,db_config)

inspector = DVDRentalInspector()
schema_info = inspector.get_schema_for_prompt()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class QueryState(TypedDict):
    """State management for query processing"""
    messages: Annotated[List[Any], add]  # Accumulates messages
    question: str
    sql_query: str
    error: Optional[str]
    context: Dict
    execution_history: Annotated[List[Dict], add]  # Accumulates history
    query_result: Optional[List[Dict]]
    response: Optional[str]  # Add the new field in QueryState
    recovery_attempts: int  # Add the new field in QueryState

def generate_sql(state: QueryState) -> QueryState:
    """Generate PostgreSQL query from natural language"""
    try:
        #Generate SQL prompt
        system_prompt = f"""you are a database expert in PostgreSQL. Generate a SQL query for the DVD rental database.
        Your task is to convert natural language questions into SQL queries.
        Do not make any assumptions about the data in the database. Always refer to the schema.
        Schema: {schema_info}
        Return only the SQL query without any explanations or markdown."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"])
        ]
    
        sql_query = llm.invoke(messages)
        sql_query = sql_query.content.replace('```sql', '').replace('```', '').strip()
        
        return {
            **state,
            "sql_query": sql_query,
            "messages": [SystemMessage(content=f"Generated SQL Query:\n{sql_query}")],
            "execution_history": [{
                "step": "generate_sql",
                "output": sql_query,
                "timestamp": datetime.now().isoformat()
            }]
        }

    except Exception as e:
        error_msg = f"Failed to generate SQL: {str(e)}"
        return {
            **state,
            "error": error_msg,
            "messages": [SystemMessage(content=f"Error: {error_msg}")],
            "execution_history": [{
                "step": "generate_sql",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }]
        }

def execute_sql(state: QueryState) -> QueryState:
    """Execute the SQL query and return results"""
    recovery = 0
    try:
        sql_query = state.get("sql_query", "").strip()
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            results = cur.fetchall()

            if len(results) == 0:
                recovery += 1
                recovered_query, suggestions = extractor.recover_query(sql_query)
                cur.execute(recovered_query)
                results = cur.fetchall()
                print(suggestions)

             # Convert Decimal to float for JSON serialization
            for row in results:
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        row[key] = float(value)
            
            return {
                **state,
                "query_result": results,
                "execution_history": [{
                    "step": "execute_sql",
                    "output": f"Query executed successfully. {len(results)} rows returned.",
                    "recovered_query": recovered_query if recovery else None,
                    "timestamp": datetime.now().isoformat()
                }],
                "messages": [SystemMessage(content=f"Query executed successfully. Found {len(results)} results.")]
            }
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        return {
            **state,
            "error": error_msg,
            "query_result": [],  # Initialize with empty list instead of None
            "execution_history": [{
                "step": "execute_sql",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }],
            "messages": [SystemMessage(content=f"Error: {error_msg}")]
        }
    finally:
        if 'conn' in locals() and conn:
            conn.close()


def recover_sql(state: QueryState) -> QueryState:
    """Attempt to fix SQL errors by analyzing the error message"""
    try:
        # Create prompt for SQL recovery
        system_prompt = f"""You are a database expert in PostgreSQL. Your task is to fix SQL queries 
        that have errors. Analyze the error message and the original query, then provide a corrected 
        version. 

        When fixing queries:
        1. Analyze the error message carefully
        2. Do not make any assumptions about the data in the database. Always consider the schema and data types of columns
        3. For "top N per group" queries, use window functions like ROW_NUMBER()
        4. film.rating is an MPAA rating (text values like 'PG', 'R') and CANNOT be averaged
        5. Ensure aggregation functions match column data types
        
        Database Schema:
        {schema_info}
        
        Return only the corrected SQL query without any explanations or markdown."""

         # Format execution history into a readable string
        history = ""
        if state.get("execution_history"):
            history = "\nPrevious attempts:\n"
            for entry in state["execution_history"]:
                if "output" in entry:  # SQL correction attempt
                    history += f"- Recovery: {entry.get('output', '')}\n"
                elif "error" in entry:  # Execution error
                    history += f"- Error: {entry.get('error', '')}\n"


        context = {
            "original_query": state["sql_query"],
            "error_message": state["error"],
            "original_question": state["question"] 
        }

        prompt = f"""
        original question: {context['original_question']}
        Original Query: {context['original_query']}
        Error Message: {context['error_message']}
        {history}
 
        Please provide a corrected SQL query that resolves this error."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        sql_query = llm.invoke(messages)
        sql_query = sql_query.content.replace('```sql', '').replace('```', '').strip()
        
        return {
            **state,  
            "sql_query": sql_query,
            "error": None,  
            "execution_history": [{
                "step": "recover_sql",
                "output": f"SQL Query corrected based on error: {state['error']}",
                "timestamp": datetime.now().isoformat()
            }],
            "messages": [SystemMessage(content=f"SQL Query corrected:\n{sql_query}")],
            "recovery_attempts": state.get("recovery_attempts", 0) + 1
        }

    except Exception as e:
        error_msg = f"Failed to recover SQL: {str(e)}"
        return {
            **state,  
            "error": error_msg,
            "execution_history": [{
                "step": "recover_sql",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }],
            "messages": [SystemMessage(content=f"Error: {error_msg}")],
            "recovery_attempts": state.get("recovery_attempts", 0) + 1 
        }


def generate_response(state: QueryState) -> QueryState:
    """Generate a natural language response from SQL results"""
    try:
        if not state.get("query_result"):
            return {
                **state,
                "response": "No results found for your query.",
                "execution_history": state.get("execution_history", []) + [{
                    "step": "generate_response",
                    "output": "No results to summarize",
                    "timestamp": datetime.now().isoformat()
                }]
            }

        system_prompt = """You are a helpful database analyst. Your task is to summarize SQL query results 
        in natural language. Focus on key insights and patterns in the data. Be concise but informative."""

        context = {
            "question": state["question"],
            "results": json.dumps(state["query_result"], indent=2)
        }

        prompt = f"""
        Original Question: {context['question']}
        Query Results: {context['results']}
        
        Please provide a natural language summary of these results, highlighting key insights."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        
        return {
            **state,
            "response": response.content,
            "execution_history": state.get("execution_history", []) + [{
                "step": "generate_response",
                "output": "Generated natural language response",
                "timestamp": datetime.now().isoformat()
            }]
        }

    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        return {
            **state,
            "error": error_msg,
            "execution_history": state.get("execution_history", []) + [{
                "step": "generate_response",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error: {error_msg}")
            ]
        }


def route_after_recovery(state: QueryState):
    """Route after recovery based on attempt count"""
    if state.get("recovery_attempts", 0) >= 3:
        return END
    return "execute_sql"

def route_by_error(state: QueryState):
    """Route based on whether there's an error and within attempt limit"""
    if state.get("error") is not None and state.get("recovery_attempts", 0) < 3:
        return "recover_sql"
    elif state.get("error") is None:
        return "generate_response"
    return END

def create_workflow() -> StateGraph:
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("recover_sql", recover_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Create flow
    workflow.add_edge("generate_sql", "execute_sql")
    
    # Add conditional edges from execute_sql based on error and attempts
    workflow.add_conditional_edges(
        "execute_sql",
        route_by_error,
        {
            "recover_sql": "recover_sql",
            "generate_response": "generate_response",
            END: END
        }
    )
    
    # Add conditional edges from recover_sql based on attempts
    workflow.add_conditional_edges(
        "recover_sql",
        route_after_recovery,
        {
            "execute_sql": "execute_sql",
            END: END
        }
    )
    
    # Add edge from generate_response to END
    workflow.add_edge("generate_response", END)
    
    workflow.set_entry_point("generate_sql")
    
    return workflow.compile()

if __name__ == "__main__":
    # Initialize app
    app = create_workflow()
    
    # Initial state
    config = {
        "question": "What are the top 5 most rented movies in each category, including their rental count and average rating?",
        "sql_query": "",
        "error": None,
        "context": {},
        "execution_history": [],
        "messages": [],
        "query_result": None,
        "response": None
    }
    
    # Run workflow
    final_state = app.invoke(config)
    
    # Print results
    print("\nQuestion:", final_state["question"])
    print("\nGenerated SQL:")
    print(final_state.get("sql_query", "No SQL generated"))
    
    if final_state.get("error"):
        print("\nFinal Error:", final_state.get("error"))
    else:
        print("\nQuery Results:")
        results = final_state.get("query_result", [])
        if results:
            print(json.dumps(results, indent=2))
        else:
            print("No results found")
    print("\nGenerated Response:")  
    print(final_state.get("response", "No response generated"))
    print("\nExecution History:")
    for entry in final_state.get("execution_history", []):
        print(f"\nStep: {entry['step']}")
        if "error" in entry:
            print(f"Error: {entry['error']}")
        else:
            print(f"Output: {entry['output']}")
        print(f"Timestamp: {entry['timestamp']}")
        
    print("\nSystem Messages:")
    for message in final_state.get("messages", []):
        print(message.content)