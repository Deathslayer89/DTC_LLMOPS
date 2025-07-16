import subprocess
import json
import sys

class MCPClient:
    def __init__(self, command):
        self.command = command
        self.process = None

    def start_server(self):
        """Start the MCP server process"""
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )

    def _send_request(self, request):
        """Send a JSON-RPC request to the server"""
        if self.process is None:
            raise RuntimeError("Server not started")
        
        request_json = json.dumps(request) + '\n'
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        return None

    def initialize(self):
        """Initialize the MCP session"""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        return self._send_request(init_request)

    def initialized(self):
        """Send initialized notification"""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        self.process.stdin.write(json.dumps(notification) + '\n')
        self.process.stdin.flush()

    def get_tools(self):
        """Get list of available tools"""
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        return self._send_request(tools_request)

    def call_tool(self, name, arguments):
        """Call a specific tool"""
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        return self._send_request(call_request)

    def close(self):
        """Close the MCP client"""
        if self.process:
            self.process.terminate()
            self.process.wait()


class MCPTools:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.tools = None
    
    def get_tools(self):
        if self.tools is None:
            mcp_tools = self.mcp_client.get_tools()
            self.tools = self.convert_tools_list(mcp_tools)
        return self.tools

    def convert_tools_list(self, mcp_tools):
        """Convert MCP tools format to OpenAI format"""
        converted_tools = []
        if 'result' in mcp_tools and 'tools' in mcp_tools['result']:
            for tool in mcp_tools['result']['tools']:
                converted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool['description'],
                        "parameters": tool['inputSchema']
                    }
                }
                converted_tools.append(converted_tool)
        return converted_tools

    def function_call(self, tool_call_response):
        function_name = tool_call_response.name
        arguments = json.loads(tool_call_response.arguments)

        result = self.mcp_client.call_tool(function_name, arguments)

        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": json.dumps(result, indent=2),
        }


if __name__ == "__main__":
    # Test the MCP client
    client = MCPClient(["python3", "weather_server.py"])
    
    try:
        client.start_server()
        print("Server started")
        
        # Initialize
        init_result = client.initialize()
        print("Initialize result:", init_result)
        
        # Send initialized notification
        client.initialized()
        print("Initialized notification sent")
        
        # Get tools
        tools = client.get_tools()
        print("Available tools:", json.dumps(tools, indent=2))
        
        # Test calling get_weather for Berlin
        berlin_weather = client.call_tool("get_weather", {"city": "Berlin"})
        print("Berlin weather:", berlin_weather)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
