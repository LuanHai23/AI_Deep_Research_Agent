from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
from groq import Groq
from src.core.llm_client import LLMClient
from src.tools.search import search_academic_papers, tools as search_tools
load_dotenv()
#1. Load environment variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = 'llama-3.3-70b-versatile'
llm = LLMClient(Groq(api_key=GROQ_API_KEY), model)

#2. Check if API key is set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

#3. Initialize FastAPI app 
app = FastAPI(title="AI Research Assistant API", description="API for searching academic papers using an LLM-powered agent.")

class ResearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

#4. Define the research endpoint
@app.post("/research")
async def search_papers(query_request: ResearchRequest):
    print(f"Received research request with query: '{query_request.query}' and max_results: {query_request.max_results}")
    #4.1 Run the conversation with the LLM client to get research results
    try:
        report = llm.run_conversation(user_query=query_request.query, tool_definitions=search_tools, available_functions={"search_academic_papers": search_academic_papers})
        return {
            "status": "success",
            "query": query_request.query,
            "max_results": query_request.max_results,
            "results": report
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

