from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple
import uvicorn
from main import create_workflow

app = FastAPI(
    title="SQL Query API",
    description="API for converting natural language questions to SQL and getting answers",
    version="1.0.0"
)

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    question: str
    sql_query: str
    answer: str

@app.post("/sql", response_model=Answer)
async def sql(question: Question):
    try:
        # Initialize the workflow
        workflow = create_workflow()
        
        # Create initial state
        initial_state = {
            "messages": [],
            "question": question.text,
            "sql_query": "",
            "error": None,
            "context": {},
            "execution_history": [],
            "query_result": None,
            "response": None,
            "recovery_attempts": 0
        }
        
        # Run the workflow
        final_state = workflow.invoke(initial_state)
        
        return {
            "question": question.text,
            "sql_query": final_state["sql_query"],
            "answer": final_state["response"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("sql_endpoint:app", host="0.0.0.0", port=8001, reload=True)
