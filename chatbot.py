from main import create_workflow
from langchain.schema import SystemMessage, HumanMessage
import json
from datetime import datetime
import os

def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def log_interaction(question, sql_query, response):
    """Log each interaction to a file"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"chat_log_{datetime.now().strftime('%Y%m%d')}.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Generated SQL: {sql_query}\n")
        f.write(f"Response: {response}\n")
        f.write(f"{'='*50}\n")


def run_chatbot():
    """Run an interactive chatbot interface for SQL queries"""
    app = create_workflow()
    print("Welcome to SQL Assistant! Ask me questions about your database.")
    print("Type 'q' to quit.\n")
    
    # Keep track of last 5 messages
    messages = []
    
    while True:
        # Get user input
        question = input("\nYour question: ").strip()
        if question.lower() == 'q':
            print("Goodbye!")
            break
            
        # Build context from previous messages
        context = ""
        if messages:
            context = "Previous context:\n"
            for msg in messages[-5:]:
                context += f"Q: {msg['question']}\n"
                context += f"SQL: {msg['sql']}\n"
                context += f"A: {msg['response']}\n\n"
            context += "Current question: "
        
        # Create initial state with context
        config = {
            "question": context + question if context else question,
            "sql_query": "",
            "error": None,
            "context": {},
            "execution_history": [],
            "messages": [],
            "query_result": None,
            "response": None,
            "recovery_attempts": 0
        }
        
        try:
            # Run workflow
            final_state = app.invoke(config)
            
            # Get the actual executed query from execution history
            executed_query = None
            for step in reversed(final_state.get("execution_history", [])):
                if step.get("step") == "execute_sql":
                    executed_query = step.get("recovered_query") or final_state.get("sql_query")
                    break
            
            # Store this interaction
            messages.append({
                "question": question,
                "sql": executed_query,
                "response": final_state.get("response")
            })

            # Log the interaction
            log_interaction(
                question=question,
                sql_query=executed_query or "No SQL generated",
                response=final_state.get("response", "No response generated")
            )
            
            # Print response
            if final_state.get("error"):
                print("\nSorry, I encountered an error:", final_state["error"])
            else:
                print("\nExecuted SQL Query:")
                print(executed_query or "No SQL generated")
                
                print("\nResponse:")
                print(final_state.get("response", "No response generated"))
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    run_chatbot()