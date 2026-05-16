import json
import requests
from dotenv import load_dotenv
import os
load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def search_academic_papers(query, max_results=5):
    print(f"Searching for academic papers with query: '{query}' and max_results: {max_results}")
    mock_results = [
        {
            "title": f"Paper {i+1} on {query}",
            "authors": [f"Author {j+1}" for j in range(3)],
            "abstract": f"This is a mock abstract for paper {i+1} related to {query}.",
            "url": f"https://arxiv.org/abs/{1000 + i}",
            "publication_year": 2020 + i
        }
        for i in range(max_results)
    ]
    return mock_results

tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_academic_papers",
                    "description": "Search for scientific research articles on academic databases such as arXiv or PubMed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Use in-depth search keywords to find articles (e.g., 'Autonomous AI agents architectures')."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "The maximum number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]