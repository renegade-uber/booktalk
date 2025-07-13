# BookTalk
BookTalk is an AI-powered book exploration tool that allows you to have conversations with your books.

## Overview
BookTalk allows you to have conversations with your books. Upload PDF or text files and ask questions about their content using semantic search and AI-powered answers.

## Features
- Load and process PDF and TXT files
- Semantic search using vector embeddings
- AI-powered answers using OpenAI's GPT models
- Source references for answer verification
- Web interface using Chainlit

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/booktalk.git
cd booktalk

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Terminal Application
Run the application in terminal mode:

```bash
python booktalk.py
```

### Web Interface (Chainlit)
Run the application with the Chainlit web interface:

```bash
chainlit run booktalk.py
```

This will start a web server and open the BookTalk interface in your browser.

## Book Storage
By default, BookTalk looks for books in `~/Documents/books`. Place your PDF or TXT files in this directory to make them available to the application.
