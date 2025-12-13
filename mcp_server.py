
import sys
import json
import logging
from rag_engine import RAGEngine
# Minimal MCP-like implementation for stdio
# Since full MCP SDK might require specific transport setup, we'll build a simple JSON-RPC style handler
# that can be adapted or reused.

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_server")

class MentalHealthMCPServer:
    def __init__(self):
        self.rag = RAGEngine()
        
    def list_tools(self):
        return [
            {
                "name": "search_knowledge_base",
                "description": "Search the local mental health knowledge base and web for information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's question or search term"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "assess_risk",
                "description": "Assess basic mental health risk based on scores.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stress_score": {"type": "integer"},
                        "sleep_hours": {"type": "number"}
                    },
                    "required": ["stress_score"]
                }
            }
        ]

    def call_tool(self, name, arguments):
        if name == "search_knowledge_base":
            query = arguments.get("query")
            results = self.rag.query(query)
            # Format results for LLM consumption
            text = "Found the following information:\n\n"
            for r in results:
                text += f"- [{r['source']}] {r['title']}: {r['content']}\n"
            return [{"type": "text", "text": text}]
            
        elif name == "assess_risk":
            score = arguments.get("stress_score")
            sleep = arguments.get("sleep_hours", 7)
            risk = "Low"
            if score > 20 or sleep < 6:
                risk = "High"
            elif score > 10:
                risk = "Moderate"
            return [{"type": "text", "text": f"Calculated Risk Level: {risk}"}]
            
        else:
            raise ValueError(f"Unknown tool: {name}")

    def run_stdio(self):
        """Run JSON-RPC over Stdio for MCP clients"""
        # Simple loop to read JSON-RPC lines
        for line in sys.stdin:
            try:
                request = json.loads(line)
                # Handle JSON-RPC / MCP specific messages
                # This is a simplified "fake" MCP for demonstration if actual SDK isn't fully piped
                # But typically clients send {"method": "tools/list", ...}
                
                method = request.get("method")
                req_id = request.get("id")
                
                response = {"jsonrpc": "2.0", "id": req_id}
                
                if method == "tools/list":
                    response["result"] = {"tools": self.list_tools()}
                    
                elif method == "tools/call":
                    params = request.get("params", {})
                    name = params.get("name")
                    args = params.get("arguments", {})
                    result = self.call_tool(name, args)
                    response["result"] = {"content": result}
                
                else:
                    # Echo or ignore check
                    continue
                    
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Error processing line: {e}")

if __name__ == "__main__":
    server = MentalHealthMCPServer()
    # If run directly often effectively tests the RAG init
    print("MCP Server Initialized. Ready for queries (Stdio Mode).", file=sys.stderr)
    # server.run_stdio() # Uncomment to run in stdio mode
