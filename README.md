# Text to SQL Chatbot

A conversational AI chatbot that converts natural language text into SQL queries.

## Project Structure

```
txt_sql_final/
├── database/              # Database related files
├── logs/                  # Application logs
├── .env                   # Environment variables, save your OPENAI_API_KEY=<YOUR_API_KEY>
├── .gitignore            # Git ignore rules
├── chatbot.py            # Core chatbot functionality
├── db_inspector.py       # Database inspection utilities
├── gradio_app.py         # Gradio web interface
├── main.py               # Main application logic
└── query_patterns.py     # SQL query pattern definitions
```

## Running the Application

### Running  with Gradio

Start the Gradio web interface:
```bash
python gradio_app.py
```

This will launch a web interface where you can:
- Input natural language queries
- View generated SQL queries
- Interact with the chatbot

### Running on the Command Line

To run the main application:
```bash
python chatbot.py
```

## Features

- Natural language to SQL conversion
- Intelligent similarity matching for handling variations in questions

