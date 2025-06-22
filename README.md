# RAG Comparison API

A FastAPI-based application that compares Retrieval-Augmented Generation (RAG) performance between Redis Semantic Search and Elasticsearch Dense Vector Search using OpenAI embeddings and LangChain.

## Features

- **Dual Vector Store Support**: Compare Redis and Elasticsearch vector databases
- **Semantic Search**: Powered by OpenAI embeddings (text-embedding-ada-002)
- **Conversational Memory**: Redis-backed chat history with session management
- **RESTful API**: FastAPI endpoints for individual and comparative searches
- **Real-time Performance Metrics**: Response time tracking for both systems
- **CORS Support**: Cross-origin requests enabled for web interfaces

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   OpenAI LLM     │────│  Vector Stores  │
│                 │    │   (GPT-4.1)      │    │                 │
│ • REST Endpoints│    │                  │    │ • Redis         │
│ • CORS Support  │    │ • Embeddings     │    │ • Elasticsearch │
│ • Session Mgmt  │    │ • Chat Completion│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   LangChain      │
                    │                  │
                    │ • Agent Executor │
                    │ • Memory Buffer  │
                    │ • Retriever Tools│
                    └──────────────────┘
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Redis instance (local or cloud)
- Elasticsearch instance (local or cloud)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-comparison-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment setup**
Create a `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration
REDIS_URL=redis://default:password@host:port

# Elasticsearch Configuration
ES_CLOUD_ID=your_elasticsearch_cloud_id
ES_API_KEY=your_elasticsearch_api_key
```

## Configuration

### Redis Setup
The application expects a Redis vector store with:
- Index name: `test2`
- Pre-populated with document embeddings
- URL format: `redis://[username]:[password]@[host]:[port]`

### Elasticsearch Setup
The application expects an Elasticsearch instance with:
- Index name: `search-rnpy`
- Pre-populated with document embeddings
- Cloud ID and API key for authentication

### Document Preparation
Before running the API, ensure your vector stores are populated with documents. The commented code in the main file shows examples of how to create and store embeddings:

```python
# For Redis
vectorstore = RedisVectorStore.from_documents(
    documents=texts,
    embedding=embedding,
    index_name="test2",
    redis_url=redis_url
)

# For Elasticsearch
vector_db = ElasticsearchStore.from_documents(
    texts,
    embedding=embedding,
    index_name="search-rnpy",
    es_cloud_id="your_cloud_id",
    es_api_key="your_api_key"
)
```

## Usage

### Starting the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Web Interface**: `http://localhost:8000/` (requires `interface.html`)

### API Endpoints

#### Health Check
```http
GET /health
```
Returns the status of all components (Redis, Elasticsearch, LLM).

#### Redis Semantic Search
```http
POST /search/redis
Content-Type: application/json

{
    "question": "Your question here",
    "session_id": "optional_session_id"
}
```

#### Elasticsearch Search
```http
POST /search/elasticsearch
Content-Type: application/json

{
    "question": "Your question here",
    "session_id": "optional_session_id"
}
```

#### Compare Both Methods
```http
POST /search/compare
Content-Type: application/json

{
    "question": "Your question here",
    "session_id": "optional_session_id"
}
```

#### Session Management
```http
# Get session history
GET /session/{session_id}/history

# Clear session history
DELETE /session/{session_id}/history
```

### Response Format

#### Individual Search Response
```json
{
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of artificial intelligence...",
    "source": "Redis Semantic Search",
    "response_time": 1.234,
    "session_id": "default_session",
    "status": "success"
}
```

#### Comparison Response
```json
{
    "question": "What is machine learning?",
    "redis_response": { ... },
    "elasticsearch_response": { ... }
}
```
## Features in Detail

### Conversational Memory
- Redis-backed chat history storage
- Configurable window memory (default: 10 messages)
- Session-based conversation tracking
- Automatic memory management

### Performance Monitoring
- Response time tracking for each query
- Comparative performance metrics
- Error handling and status reporting

### LangChain Integration
- OpenAI Tools Agent for advanced reasoning
- Retriever tools for vector store queries
- Prompt templates for consistent responses
- Agent executor with memory integration

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Redis/Elasticsearch credentials in `.env`
   - Check network connectivity to cloud services
   - Ensure vector stores are properly initialized

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

3. **OpenAI API Issues**
   - Verify API key validity
   - Check usage limits and billing status

4. **Memory Errors**
   - Clear session history: `DELETE /session/{session_id}/history`
   - Restart the application

### Debugging
Enable verbose logging by setting environment variable:
```bash
export LANGCHAIN_DEBUG=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Create an issue in the repository
