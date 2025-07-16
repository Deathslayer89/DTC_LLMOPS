# Agents Homework

## 📁 Files Structure

- [`weather_server.py`](weather_server.py) - MCP server implementation with weather functions
- [`mcp_client.py`](mcp_client.py) - Synchronous MCP client for integration
- [`chat_assistant.py`](chat_assistant.py) - Chat assistant framework (downloaded from course repo)
- [`test_mcp_client.py`](test_mcp_client.py) - test script

### Q1. Define function description

**Question:** Fill in the missing parts for the `get_weather` function description.

**Answer:** `city`

**description:**
```python
get_weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather temperature for a specified city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get weather for"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }
}
```

### Q2. Adding another tool

**Question:** Write a description for the `set_weather` function.

**Answer:**
```python
set_weather_tool = {
    "type": "function",
    "name": "set_weather",
    "description": "Set the temperature for a specified city in the weather database",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to set weather for"
            },
            "temp": {
                "type": "number",
                "description": "The temperature value to set for the city"
            }
        },
        "required": ["city", "temp"],
        "additionalProperties": False
    }
}
```

### Q3. Install FastMCP

**Question:** What's the version of FastMCP you installed?

**Answer:** `2.10.5`

### Q4. Simple MCP Server

**Question:** What transport does the MCP server use?

**Answer:** `stdio`

**Server output:**
```
Starting MCP server 'Demo 🚀' with transport 'stdio'
```

### Q5. Protocol

**Question:** What response do you get when asking for Berlin weather?

**Answer:**
```json
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "20.0"}], "structuredContent": {"result": 20.0}, "isError": false}}
```

### Q6. Client

**Question:** How does the result look when getting available tools?

**Answer:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Retrieves the temperature for a specified city.\n\nParameters:\n    city (str): The name of the city for which to retrieve weather data.\n\nReturns:\n    float: The temperature associated with the city.",
        "inputSchema": {
          "properties": {
            "city": {
              "title": "City",
              "type": "string"
            }
          },
          "required": [
            "city"
          ],
          "type": "object"
        },
        "outputSchema": {
          "properties": {
            "result": {
              "title": "Result",
              "type": "number"
            }
          },
          "required": [
            "result"
          ],
          "title": "_WrappedResult",
          "type": "object",
          "x-fastmcp-wrap-result": true
        }
      },
      {
        "name": "set_weather",
        "description": "Sets the temperature for a specified city.\n\nParameters:\n    city (str): The name of the city for which to set the weather data.\n    temp (float): The temperature to associate with the city.\n\nReturns:\n    str: A confirmation string 'OK' indicating successful update.",
        "inputSchema": {
          "properties": {
            "city": {
              "title": "City",
              "type": "string"
            },
            "temp": {
              "title": "Temp",
              "type": "number"
            }
          },
          "required": [
            "city",
            "temp"
          ],
          "type": "object"
        }
      }
    ]
  }
}
```

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install fastmcp
   ```

2. **Run the MCP server:**
   ```bash
   python3 weather_server.py
   ```

3. **Test the MCP client:**
   ```bash
   python3 test_mcp_client.py
   ```

## 🔧 Implementation Details

### [`weather_server.py`](weather_server.py)
- **FastMCP server** with two weather functions
- `get_weather(city: str) -> float` - Retrieves temperature for a city
- `set_weather(city: str, temp: float) -> str` - Sets temperature for a city
- Uses proper docstrings for automatic tool registration

### [`mcp_client.py`](mcp_client.py)
- **Synchronous MCP client** implementation
- JSON-RPC protocol handling with proper handshake
- Tool listing and calling functionality
- Integration wrapper for chat assistants
- Error handling and process management

### [`chat_assistant.py`](chat_assistant.py)
- Chat assistant framework from the course repository
- Supports function calling integration
- Can be extended to work with MCP tools
