from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables
load_dotenv()

# Updated imports based on your original code
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_community.vectorstores import Redis
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_elasticsearch import ElasticsearchStore

# FastAPI app initialization
app = FastAPI(title="RAG Comparison API", description="Compare Redis Semantic Search vs Elasticsearch RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
# Create a 'static' directory in your project root and put your HTML files there
# app.mount("/static", StaticFiles(directory="static"), name="static")



# # Redis configuration
# index_name = "test2"
# redis_url = "redis://default:1PhTVfuSdeBKSvKEqcmBQRGFzEXwza9z@redis-18318.c281.us-east-1-2.ec2.redns.redis-cloud.com:18318"

# # Commented out code for creating and storing embeddings in Redis
# vectorstore = RedisVectorStore.from_documents(
#     documents=texts,
#     embedding=embedding,
#     index_name=index_name,
#     redis_url=redis_url
# )


# # Commented out code for creating and storing embeddings in Elastic cloud

# vector_db=ElasticsearchStore.from_documents(
#     texts,
#     embedding=embedding,
#     index_name="search-rnpy",
#     es_cloud_id="3b8a5223f6d94c199d213c7fe51f4826:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbSQwNTY4MzVkNzkyZmQ0MDljOTkzODczYWUxODUxMWY3NiRmYzE1MDY5OTUzOGE0MGRiOWI0N2VhOTRhMWNmZTEwZQ==",
#     es_api_key="MzBrb2w1Y0I2d1lBNjg1S2pkdG06dk81YWpIWFpYNmJwN3dtRHFsNnFzdw=="
# )


# Serve the main HTML interface at root path
@app.get("/")
async def read_index():
    return FileResponse('interface.html')

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"

class RAGResponse(BaseModel):
    question: str
    answer: str
    source: str
    response_time: float
    session_id: str
    status: str

class ComparisonResponse(BaseModel):
    question: str
    redis_response: RAGResponse
    elasticsearch_response: RAGResponse

# Global variables for vector stores
redis_vectorstore = None
elastic_vectorstore = None
embedding = None
llm = None

# Initialize components
def initialize_components():
    global redis_vectorstore, elastic_vectorstore, embedding, llm
    
    # OpenAI embeddings setup
    embedding = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        model="text-embedding-ada-002"
    )
    
    # Redis configuration
    index_name = "test2"
    # redis_url = "redis://default:1PhTVfuSdeBKSvKEqcmBQRGFzEXwza9z@redis-18318.c281.us-east-1-2.ec2.redns.redis-cloud.com:18318"
    redis_url = os.getenv("REDIS_URL")
    
    # Initialize Redis vector store
    try:
        redis_vectorstore = Redis.from_existing_index(
            embedding=embedding,
            index_name=index_name,
            redis_url=redis_url,
            schema=None
        )
        print("✅ Redis vector store initialized successfully")
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")
        redis_vectorstore = None
    
    # Initialize Elasticsearch vector store
    try:
        elastic_vectorstore = ElasticsearchStore(
            embedding=embedding,
            index_name="search-rnpy",
            es_cloud_id=os.getenv("ES_CLOUD_ID"),
            es_api_key=os.getenv("ES_API_KEY"),
        )
        print("✅ Elasticsearch vector store initialized successfully")
    except Exception as e:
        print(f"❌ Error connecting to Elasticsearch: {e}")
        elastic_vectorstore = None
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )
    print("✅ LLM initialized successfully")

# Shared prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Answer:."""),
    MessagesPlaceholder("{chat_history}", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

def get_chat_memory(session_id: str):
    """Initialize Redis chat memory for conversation history"""
    redis_url = "redis://default:1PhTVfuSdeBKSvKEqcmBQRGFzEXwza9z@redis-18318.c281.us-east-1-2.ec2.redns.redis-cloud.com:18318"
    
    try:
        message_history = RedisChatMessageHistory(
            url=f"{redis_url}/0",
            session_id=session_id
        )
        
        window_memory = ConversationBufferWindowMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,
            k=10,
        )
        return window_memory
    except Exception as e:
        print(f"Error initializing chat memory: {e}")
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,
            k=10,
        )

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    initialize_components()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "redis_available": redis_vectorstore is not None,
        "elasticsearch_available": elastic_vectorstore is not None,
        "llm_available": llm is not None
    }

# Redis semantic search endpoint
@app.post("/search/redis", response_model=RAGResponse)
async def redis_semantic_search(request: QueryRequest):
    if not redis_vectorstore:
        raise HTTPException(status_code=503, detail="Redis vector store is not available")
    
    start_time = time.time()
    
    try:
        # Create retriever tool
        redis_rag_tool = create_retriever_tool(
            redis_vectorstore.as_retriever(search_kwargs={"k": 4}),
            "get_content_from_redis_vector_store",
            "Use this tool to retrieve relevant content from the Redis vector store based on the user's query."
        )
        
        # Create agent
        agent = create_openai_tools_agent(llm, [redis_rag_tool], prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            memory=get_chat_memory(f"redis_{request.session_id}"),
            tools=[redis_rag_tool],
            verbose=True
        )
        
        # Execute query
        result = agent_executor.invoke({"input": request.question})
        response_time = time.time() - start_time
        
        return RAGResponse(
            question=request.question,
            answer=result["output"],
            source="Redis Semantic Search",
            response_time=response_time,
            session_id=request.session_id,
            status="success"
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        return RAGResponse(
            question=request.question,
            answer=f"Error in Redis semantic search: {str(e)}",
            source="Redis Semantic Search",
            response_time=response_time,
            session_id=request.session_id,
            status="error"
        )

# Elasticsearch search endpoint
@app.post("/search/elasticsearch", response_model=RAGResponse)
async def elasticsearch_search(request: QueryRequest):
    if not elastic_vectorstore:
        raise HTTPException(status_code=503, detail="Elasticsearch vector store is not available")
    
    start_time = time.time()
    
    try:
        # Create retriever tool
        elastic_rag_tool = create_retriever_tool(
            elastic_vectorstore.as_retriever(search_kwargs={"k": 4}),
            "get_content_from_elastic_vector_store",
            "Use this tool to retrieve relevant content from the Elasticsearch vector store based on the user's query."
        )
        
        # Create agent
        agent = create_openai_tools_agent(llm, [elastic_rag_tool], prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            memory=get_chat_memory(f"elastic_{request.session_id}"),
            tools=[elastic_rag_tool],
            verbose=True
        )
        
        # Execute query
        result = agent_executor.invoke({"input": request.question})
        response_time = time.time() - start_time
        
        return RAGResponse(
            question=request.question,
            answer=result["output"],
            source="Elasticsearch Dense Vector",
            response_time=response_time,
            session_id=request.session_id,
            status="success"
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        return RAGResponse(
            question=request.question,
            answer=f"Error in Elasticsearch search: {str(e)}",
            source="Elasticsearch Dense Vector",
            response_time=response_time,
            session_id=request.session_id,
            status="error"
        )

# Compare both search methods
@app.post("/search/compare", response_model=ComparisonResponse)
async def compare_searches(request: QueryRequest):
    # Run both searches concurrently
    redis_task = redis_semantic_search(request)
    elasticsearch_task = elasticsearch_search(request)
    
    # Wait for both to complete
    redis_result, elasticsearch_result = await asyncio.gather(
        redis_task, elasticsearch_task, return_exceptions=True
    )
    
    # Handle any exceptions
    if isinstance(redis_result, Exception):
        redis_result = RAGResponse(
            question=request.question,
            answer=f"Error: {str(redis_result)}",
            source="Redis Semantic Search",
            response_time=0.0,
            session_id=request.session_id,
            status="error"
        )
    
    if isinstance(elasticsearch_result, Exception):
        elasticsearch_result = RAGResponse(
            question=request.question,
            answer=f"Error: {str(elasticsearch_result)}",
            source="Elasticsearch Dense Vector",
            response_time=0.0,
            session_id=request.session_id,
            status="error"
        )
    
    return ComparisonResponse(
        question=request.question,
        redis_response=redis_result,
        elasticsearch_response=elasticsearch_result
    )

# Get session history
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    try:
        memory = get_chat_memory(session_id)
        history = memory.chat_memory.messages
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": [{"type": msg.type, "content": msg.content} for msg in history[-10:]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")

# Clear session history
@app.delete("/session/{session_id}/history")
async def clear_session_history(session_id: str):
    try:
        memory = get_chat_memory(session_id)
        memory.clear()
        return {"message": f"Session {session_id} history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)