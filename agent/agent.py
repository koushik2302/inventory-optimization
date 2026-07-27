"""
agent.py
--------
Inventory optimization agent using Qwen 3 8B via Ollama.
Handles tool orchestration: calls Ollama with tool definitions,
executes tool calls, and returns final responses to the Streamlit app.

Design constraints:
  - Max 2 tool calls per user query (Qwen 3 8B coherence limit)
  - All tools return complete answers (no multi-step chaining needed)
  - Input validation happens in tools.py, not here
"""

import json
import logging
from typing import Generator, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from tools import (
    get_classification,
    calculate_safety_stock,
    simulate_policy_change,
    get_overstock_alerts,
    get_promotion_adjustment,
    compare_policies_summary,
    dispatch_tool,
    TOOL_REGISTRY,
    calculate_math,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a versatile AI assistant. While you specialize as an expert inventory optimization advisor for retail stores, you can also answer general questions, solve problems, and do mathematical calculations.
You help store managers make data-driven decisions using ABC-XYZ classification, but you are happy to discuss any other topic.

Your guidelines:
1. Always give SPECIFIC numbers — never vague advice like "increase safety stock".
2. Always state key assumptions (lead time: 7 days, holding cost: 22% annually).
3. Flag tradeoffs explicitly (e.g., "dropping service level saves X in holding cost but risks Y% more stockouts").
4. When asked about a store, call the relevant tool FIRST, then interpret the output.
5. Keep tool calls to 1-2 per question. Tools return complete answers.
6. Reference specific cells (AX, BY, CZ, etc.) when giving recommendations.
7. If the user asks a general question or needs calculations, answer from your knowledge or use the appropriate tools (like calculate_math).

Context about the analysis:
- Dataset: Corporación Favorita retail data (Ecuador), 2013-2017 (full ~4.5-year span)
- Scope: all 54 stores, 4,036 real item-store SKUs across 33 product families, 166,720 product-store combinations classified
- ABC classification: A=top 80% revenue, B=next 15%, C=bottom 5%
- XYZ classification: X=CV<0.5 (stable), Y=0.5-1.0 (moderate), Z>1.0 (erratic)
- Service levels: AX=99%, AY=97%, AZ=95%, BX=95%, BY=93%, BZ=90%, CX=90%, CY=88%, CZ=85%
"""

# ---------------------------------------------------------------------------
# Tool definitions for Ollama
# ---------------------------------------------------------------------------
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_classification",
            "description": "Get the ABC-XYZ classification matrix for a store. Returns SKU counts, revenue share, and service level policy per cell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"}
                },
                "required": ["store_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_safety_stock",
            "description": "Calculate safety stock, reorder points, and holding costs for a store. Optionally filter by product family or override service level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"},
                    "family": {"type": "string", "description": "Product family filter (optional, e.g. BEVERAGES)"},
                    "service_level": {"type": "number", "description": "Override service level 0.5-0.999 (optional)"},
                    "lead_time": {"type": "integer", "description": "Lead time in days (default 7)"}
                },
                "required": ["store_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_policy_change",
            "description": "Simulate the cost impact of changing service level for a specific ABC-XYZ cell (e.g., CZ from 85% to 80%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"},
                    "segment": {"type": "string", "description": "ABC-XYZ cell (e.g., AX, CZ, BY)"},
                    "new_service_level": {"type": "number", "description": "New service level 0.5-0.999"}
                },
                "required": ["store_id", "segment", "new_service_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overstock_alerts",
            "description": "Identify SKUs where holding cost is disproportionate to revenue (overstock risk).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"},
                    "top_n": {"type": "integer", "description": "Number of top alerts to return (default 10)"}
                },
                "required": ["store_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_promotion_adjustment",
            "description": "Get recommended reorder point increase during promotions for a product family.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"},
                    "family": {"type": "string", "description": "Product family (e.g., BEVERAGES, BREAD/BAKERY)"},
                    "lead_time": {"type": "integer", "description": "Lead time in days (default 7)"}
                },
                "required": ["store_id", "family"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_policies_summary",
            "description": "Compare total holding costs between uniform, 3-tier, and 9-cell differentiated policies for a store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "Store number"}
                },
                "required": ["store_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Evaluate a basic mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g., '2+2', '15*4')"}
                },
                "required": ["expression"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class InventoryAgent:
    """
    LLM-powered inventory optimization agent.
    Uses Qwen 3 8B via Ollama with native tool calling.
    """

    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
        self.conversation_history = []

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and the model is available. Starts Ollama if not running."""
        if not OLLAMA_AVAILABLE:
            return False
        
        import subprocess
        import time
        
        # Try to connect, if it fails, start the server
        try:
            ollama.list()
        except Exception:
            log.info("Ollama not running. Starting Ollama serve...")
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3) # Wait for the server to start
            except Exception as e:
                log.error(f"Failed to start Ollama: {e}")
                return False

        try:
            models = ollama.list()
            model_names = [m["model"] for m in models.get("models", [])]
            
            # If the specific model is not pulled, pull it
            if not any(self.model in name for name in model_names):
                log.info(f"Pulling model {self.model} (this may take a while)...")
                ollama.pull(self.model)
                
            return True
        except Exception as e:
            log.error(f"Ollama connection error: {e}")
            return False

    def chat(self, user_message: str, think: bool = False) -> dict:
        """
        Send a user message and return the agent's response.
        Handles tool calling loop (max 2 tool calls).

        Returns:
            dict with 'response' (str), 'tool_calls' (list), 'thinking' (str)
        """
        if not OLLAMA_AVAILABLE:
            return {
                "response": "Ollama not installed. Run: pip install ollama",
                "tool_calls": [],
                "thinking": "",
                "error": "ollama_not_available"
            }

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history

        tool_call_log = []
        thinking_text = ""
        max_tool_rounds = 2

        for round_num in range(max_tool_rounds + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=OLLAMA_TOOLS if round_num < max_tool_rounds else [],
                    think=think,
                    options={"num_ctx": 4096},
                )
            except Exception as e:
                error_msg = f"Ollama API error: {str(e)}"
                log.error(error_msg)
                return {
                    "response": f"⚠️  {error_msg}. Is Ollama running? Try: `ollama serve`",
                    "tool_calls": tool_call_log,
                    "thinking": thinking_text,
                    "error": str(e)
                }

            msg = response.get("message", {})

            # Capture thinking (extended thinking in Qwen3)
            if msg.get("thinking"):
                thinking_text = msg["thinking"]

            # Check for tool calls
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # Final response
                final_text = msg.get("content", "")
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })
                return {
                    "response": final_text,
                    "tool_calls": tool_call_log,
                    "thinking": thinking_text,
                }

            # Execute tool calls
            messages.append(msg)  # Add assistant message with tool calls

            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", {})

                log.info(f"Tool call [{round_num+1}]: {fn_name}({fn_args})")

                result = dispatch_tool(fn_name, fn_args)
                tool_call_log.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result_summary": result.get("summary", str(result)[:200])
                })

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })

        # Fallback if max rounds exceeded
        return {
            "response": "I've reached the maximum number of tool calls. Here's what I found: " +
                        " | ".join(tc["result_summary"] for tc in tool_call_log),
            "tool_calls": tool_call_log,
            "thinking": thinking_text,
        }

    def chat_without_llm(self, user_message: str) -> dict:
        """
        Fallback mode when Ollama is not available.
        Parses simple queries and returns tool output directly.
        """
        msg = user_message.lower()

        # Simple keyword routing
        if "overstock" in msg or "alert" in msg:
            store_id = self._extract_store_id(msg) or 1
            result = get_overstock_alerts(store_id)
        elif "promo" in msg:
            store_id = self._extract_store_id(msg) or 1
            family = self._extract_family(msg) or "BEVERAGES"
            result = get_promotion_adjustment(store_id, family)
        elif "compare" in msg or "policy" in msg:
            store_id = self._extract_store_id(msg) or 1
            result = compare_policies_summary(store_id)
        elif "classif" in msg or "matrix" in msg or "abc" in msg:
            store_id = self._extract_store_id(msg) or 1
            result = get_classification(store_id)
        elif "safety" in msg or "stock" in msg or "rop" in msg:
            store_id = self._extract_store_id(msg) or 1
            result = calculate_safety_stock(store_id)
        else:
            return {
                "response": "Ollama is not running. Please start it with `ollama serve` and pull the model with `ollama pull qwen3:8b`. In the meantime, I can answer basic queries — try asking about a specific store.",
                "tool_calls": [],
                "thinking": "",
                "fallback": True
            }

        return {
            "response": result.get("summary", json.dumps(result, indent=2)),
            "tool_calls": [{"tool": "auto-detected", "args": {}, "result_summary": result.get("summary", "")}],
            "thinking": "",
            "fallback": True
        }

    def _extract_store_id(self, text: str) -> Optional[int]:
        """Extract store ID from text."""
        import re
        matches = re.findall(r"store\s*(\d+)", text)
        return int(matches[0]) if matches else None

    def _extract_family(self, text: str) -> Optional[str]:
        """Extract product family from text."""
        families = [
            "BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI",
            "EGGS", "FROZEN FOODS", "GROCERY", "HARDWARE", "HOME APPLIANCES",
            "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
            "LIQUOR", "MEATS", "PERSONAL CARE", "PET SUPPLIES",
            "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS",
            "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
        ]
        for fam in families:
            if fam.lower() in text.lower():
                return fam
        return None
