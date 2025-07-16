import mcp_client
client = mcp_client.MCPClient(['python3', 'weather_server.py'])
client.start_server()
client.initialize()
client.initialized()
print(client.get_tools())
print(client.call_tool('get_weather', {'city': 'Berlin'}))
client.close()