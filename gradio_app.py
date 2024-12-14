import gradio as gr
from main import create_workflow
from langchain.schema import SystemMessage, HumanMessage
import json
from datetime import datetime
from collections import deque
import os

def log_interaction(question, sql_query, response):
    """Log each interaction to a file"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"gradio_log_{datetime.now().strftime('%Y%m%d')}.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Generated SQL: {sql_query}\n")
        f.write(f"Response: {response}\n")
        f.write(f"{'='*50}\n")


class ChatHistory:
    def __init__(self, max_messages: int = 5):
        self.messages = deque(maxlen=max_messages)
    
    def add_message(self, question: str, sql: str, response: str):
        self.messages.append({
            "question": question,
            "sql": sql,
            "response": response
        })
    
    def get_context_string(self) -> str:
        """Get history formatted for model context"""
        context = ""
        for msg in self.messages:
            context += f"Q: {msg['question']}\nSQL: {msg['sql']}\nA: {msg['response']}\n\n"
        return context
    
    def get_display_string(self) -> str:
        """Get history formatted as plain text"""
        history = "Last 5 Interactions:\n\n"
        for i, msg in enumerate(reversed(list(self.messages)), 1):
            history += f"[{i}] Question: {msg['question']}\n"
            history += f"SQL: {msg['sql']}\n"
            history += f"Response: {msg['response']}\n"
            history += "-" * 50 + "\n\n"
        return history

def process_query(question: str, chat_history: ChatHistory) -> tuple:
    """Process a single query and return the response"""
    app = create_workflow()
    
    # Build context from history
    context = chat_history.get_context_string()
    
    # Create initial state
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
        
        # Get the executed query
        executed_query = None
        for step in reversed(final_state.get("execution_history", [])):
            if step.get("step") == "execute_sql":
                executed_query = step.get("recovered_query") or final_state.get("sql_query")
                break
        
        # Format current response
        response = f"SQL Query:\n{executed_query}\n\nResponse:\n{final_state.get('response', 'No response generated')}"
        # Log the interaction
        actual_response = final_state.get('response', 'No response generated')
        log_interaction(question, executed_query, actual_response)
        
        if not final_state.get("error"):
            # Add to history
            chat_history.add_message(
                question=question,
                sql=executed_query,
                response=final_state.get('response', 'No response generated')
            )
        else:
            response = f"Error: {final_state['error']}"
    
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    
    return response, chat_history.get_display_string(),""

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# SQL Text to SQL Assistant")
        gr.Markdown("Ask questions about DVD rental database!")
        
        # Initialize chat history
        chat_history = gr.State(ChatHistory())
        
        # Input textbox
        question = gr.Textbox(
            label="Your question",
            placeholder="Your question here...",
            lines=2
        )
        
        # Submit button
        submit_btn = gr.Button("Submit")
        
        # Output textboxes
        current_response = gr.Textbox(
            label="Current Response",
            lines=6
        )
        
        history_display = gr.Textbox(
            label="Conversation History",
            lines=15,
            max_lines=15
        )
        
        # Handle submission
        submit_btn.click(
            fn=process_query,
            inputs=[question, chat_history],
            outputs=[current_response, history_display,question]
        )
    
    return demo

# Create and launch the app
demo = create_gradio_interface()

# For local testing
if __name__ == "__main__":
    demo.launch(share = True)
else:
    # For Hugging Face deployment
    demo.launch(debug=False, share=False)
