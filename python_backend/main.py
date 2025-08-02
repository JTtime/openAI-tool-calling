import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import asyncio
from openai import AsyncOpenAI
import os



# Initialize FastAPI app
app = FastAPI(title="Product Assistant API", version="1.0.0")



# 👇 Allow requests from React frontend
origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",   # Sometimes needed
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # 👈 Frontend origins
    allow_credentials=True,
    allow_methods=["*"],             # 👈 Allow all methods: GET, POST, etc.
    allow_headers=["*"],             # 👈 Allow all headers
)
# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class Product(BaseModel):
    id: int
    title: str
    description: str
    price: float
    category: str
    brand: str
    rating: float
    stock: int
    thumbnail: str

# DummyJSON API functions
async def get_all_products() -> Dict[str, Any]:
    """Get all products from DummyJSON API"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://dummyjson.com/products")
        return response.json()

async def get_product_by_id(product_id: int) -> Dict[str, Any]:
    """Get single product by ID"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://dummyjson.com/products/{product_id}")
        return response.json()

async def search_products(query: str) -> Dict[str, Any]:
    """Search products by query"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://dummyjson.com/products/search?q={query}")
        return response.json()

async def get_category_list() -> List[str]:
    """Get all product categories"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://dummyjson.com/products/category-list")
        return response.json()

async def get_products_by_category(category: str) -> Dict[str, Any]:
    """Get products by category"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://dummyjson.com/products/category/{category}")
        return response.json()

# Helper functions for intelligent product filtering
async def find_matching_category(user_query: str, categories: List[str]) -> str:
    """Use OpenAI to find the most semantically similar category"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a category matching assistant. Given a user query and a list of product categories, return the EXACT category name that best matches the user's intent. Return only the category name, nothing else."
                },
                {
                    "role": "user",
                    "content": f"User query: '{user_query}'\n\nAvailable categories: {', '.join(categories)}\n\nReturn the exact matching category name:"
                }
            ],
            temperature=0.1,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in category matching: {e}")
        return ""

async def find_cheapest_products(products: List[Dict], limit: int = 5) -> List[Dict]:
    """Find the cheapest products from a list"""
    sorted_products = sorted(products, key=lambda x: x.get('price', float('inf')))
    return sorted_products[:limit]

async def find_most_expensive_products(products: List[Dict], limit: int = 5) -> List[Dict]:
    """Find the most expensive products from a list"""
    sorted_products = sorted(products, key=lambda x: x.get('price', 0), reverse=True)
    return sorted_products[:limit]

async def filter_products_by_price_range(products: List[Dict], min_price: float = None, max_price: float = None) -> List[Dict]:
    """Filter products by price range"""
    filtered = products
    if min_price is not None:
        filtered = [p for p in filtered if p.get('price', 0) >= min_price]
    if max_price is not None:
        filtered = [p for p in filtered if p.get('price', float('inf')) <= max_price]
    return filtered

# Tool functions for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_all_products",
            "description": "Get all products from the store",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_by_id",
            "description": "Get detailed information about a specific product by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product to retrieve"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products using a query string",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for products"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_list",
            "description": "Get all available product categories",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_products_by_category",
            "description": "Get all products from a specific category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category name to filter products by"
                    }
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_products_in_category",
            "description": "Find the cheapest products in a specific category or search query",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_or_query": {
                        "type": "string",
                        "description": "Category name or search query to find cheapest products in"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of cheapest products to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["category_or_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_expensive_products_in_category",
            "description": "Find the most expensive products in a specific category or search query",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_or_query": {
                        "type": "string",
                        "description": "Category name or search query to find most expensive products in"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of most expensive products to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["category_or_query"]
            }
        }
    }
]

# Tool execution functions
async def execute_tool_call(tool_call) -> str:
    """Execute a tool call and return the result"""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    try:
        if function_name == "get_all_products":
            result = await get_all_products()
            return json.dumps(result)
        
        elif function_name == "get_product_by_id":
            result = await get_product_by_id(arguments["product_id"])
            return json.dumps(result)
        
        elif function_name == "search_products":
            result = await search_products(arguments["query"])
            return json.dumps(result)
        
        elif function_name == "get_category_list":
            result = await get_category_list()
            return json.dumps(result)
        
        elif function_name == "get_products_by_category":
            result = await get_products_by_category(arguments["category"])
            return json.dumps(result)
        
        elif function_name == "find_cheapest_products_in_category":
            category_or_query = arguments["category_or_query"]
            limit = arguments.get("limit", 5)
            
            # First, get all categories to check if it's a valid category
            categories = await get_category_list()
            
            # Try to find matching category using semantic matching
            if category_or_query.lower() not in [cat.lower() for cat in categories]:
                matched_category = await find_matching_category(category_or_query, categories)
                if matched_category and matched_category in categories:
                    category_or_query = matched_category
            
            # Try to get products by category first
            if category_or_query in categories:
                products_data = await get_products_by_category(category_or_query)
                products = products_data.get("products", [])
            else:
                # If not a valid category, search for products
                products_data = await search_products(category_or_query)
                products = products_data.get("products", [])
            
            # Find cheapest products
            cheapest = await find_cheapest_products(products, limit)
            
            # Get detailed information for each cheapest product
            detailed_products = []
            for product in cheapest:
                detailed = await get_product_by_id(product["id"])
                detailed_products.append(detailed)
            
            return json.dumps({
                "cheapest_products": detailed_products,
                "search_method": "category" if category_or_query in categories else "search",
                "total_products_found": len(products)
            })
        
        elif function_name == "find_most_expensive_products_in_category":
            category_or_query = arguments["category_or_query"]
            limit = arguments.get("limit", 5)
            
            # First, get all categories to check if it's a valid category
            categories = await get_category_list()
            
            # Try to find matching category using semantic matching
            if category_or_query.lower() not in [cat.lower() for cat in categories]:
                matched_category = await find_matching_category(category_or_query, categories)
                if matched_category and matched_category in categories:
                    category_or_query = matched_category
            
            # Try to get products by category first
            if category_or_query in categories:
                products_data = await get_products_by_category(category_or_query)
                products = products_data.get("products", [])
            else:
                # If not a valid category, search for products
                products_data = await search_products(category_or_query)
                products = products_data.get("products", [])
            
            # Find most expensive products
            most_expensive = await find_most_expensive_products(products, limit)
            
            # Get detailed information for each expensive product
            detailed_products = []
            for product in most_expensive:
                detailed = await get_product_by_id(product["id"])
                detailed_products.append(detailed)
            
            return json.dumps({
                "most_expensive_products": detailed_products,
                "search_method": "category" if category_or_query in categories else "search",
                "total_products_found": len(products)
            })
        
        else:
            return json.dumps({"error": f"Unknown function: {function_name}"})
    
    except Exception as e:
        return json.dumps({"error": f"Error executing {function_name}: {str(e)}"})

async def generate_streaming_response(request: ChatRequest):
    """Generate streaming response using OpenAI with tool calling"""
    
    # System prompt
    system_prompt = """You are a helpful product assistant for an e-commerce store. You can help users:
    
    1. Search for products
    2. Get product details
    3. Find products by category
    4. Find cheapest or most expensive products in any category
    5. Compare products
    6. Get product recommendations
    
    You have access to a comprehensive product database with various categories like electronics, clothing, home goods, and more.
    
    When users ask for cheapest or most expensive products in a category:
    - Use the specialized functions to find cheapest/most expensive products
    - These functions will automatically handle category matching and product retrieval
    - Always provide detailed product information including price, description, and availability
    
    Be conversational, helpful, and provide detailed product information when requested.
    """
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in request.conversation_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": request.message})
    
    # First call to OpenAI with tools
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False
    )
    
    message = response.choices[0].message
    
    # Handle tool calls if present
    if message.tool_calls:
        # Add the assistant's message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        })
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_result = await execute_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Get final response with streaming
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        
        async def generate():
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return generate()
    
    else:
        # No tool calls, stream the direct response
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        
        async def generate():
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return generate()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with streaming response"""
    try:
        generator = await generate_streaming_response(request)
        return StreamingResponse(
            generator,
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Product Assistant API is running"}

# Get categories endpoint
@app.get("/categories")
async def get_categories():
    """Get all available product categories"""
    try:
        categories = await get_category_list()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)