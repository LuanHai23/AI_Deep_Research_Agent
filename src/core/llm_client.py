from groq import Groq
import os
import json
from dotenv import load_dotenv
load_dotenv()
from src.tools.search import search_academic_papers, tools as search_tools

#1. Khởi tạo client Groq với API key từ biến môi trường
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
if not client:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

model = 'llama-3.3-70b-versatile'
#2. Tạo một hàm giả lập để tìm kiếm bài báo học thuật
class LLMClient:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def run_conversation(self, user_query: str, tool_definitions, available_functions):
        messages = [
            {
                "role": "system",
                "content": "You are an expert AI Research Assistant. Use tools to find academic papers when needed."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        
        response  = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=search_tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=1000
        )

        response_message = response.choices[0].message
        print(f"Assistant: {response_message.content}")
        tool_calls = response_message.tool_calls
        print(f"Tool calls: {tool_calls}")
        print("Final Response: ", response)

        if tool_calls:
            available_functions = available_functions
            
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    query=function_args.get("query"),
                    max_results=function_args.get("max_results")
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    }
                )
            second_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return second_response.choices[0].message.content
        else:
            return response_message.content

if __name__ == "__main__":
    final_output = LLMClient(client, model).run_conversation()
    print("Final Output: ", final_output)

    
