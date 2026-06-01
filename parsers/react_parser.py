"""
parsers/react_parser.py — LLM Response Parser
===============================================

WHY THIS EXISTS:
    LLMs are probabilistic text generators — they don't reliably produce
    valid JSON. Even with explicit instructions, they will sometimes:

    1. Wrap JSON in markdown code fences: ```json {...} ```
    2. Add explanatory text before/after the JSON
    3. Use single quotes instead of double quotes
    4. Include trailing commas (invalid JSON)
    5. Omit required fields
    6. Produce partial JSON when hitting token limits
    7. Nest the action_input as a string instead of a dict

    A brittle parser that crashes on any of these turns the entire
    agent into a house of cards. This parser is designed to handle
    every failure mode gracefully.

DESIGN DECISIONS:
    1. MULTIPLE EXTRACTION STRATEGIES — Try strict JSON first, then
       progressively looser strategies (regex, bracket matching).

    2. VALIDATION LAYER — After extraction, validate that required
       fields exist and have correct types.

    3. NEVER CRASH — Always return a ParsedResponse, even if it
       contains an error. The agent loop can then decide what to do
       (retry, ask LLM to fix, give up).

    4. DEFENSIVE DEFAULTS — Missing optional fields get safe defaults
       rather than None/undefined.

HOW IT CONNECTS:
    - agent/nodes.py calls parse_react_response() on every LLM output.
    - The returned ParsedResponse determines whether to call a tool
      or emit a final answer.
    - Errors in parsing become observations that the LLM can see
      and self-correct.

SCALABILITY:
    Phase 2+ can add:
    - Pydantic-based schema validation with error details,
    - LLM retry with the parsing error as feedback,
    - structured output mode (tool_use API for Claude, function calling for GPT).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Parsed Response Model
# ===================================================================

class ParsedResponse(BaseModel):
    """
    Standardized output from the parser.

    The agent loop only interacts with THIS object — it never
    touches raw LLM strings directly.
    """
    thought: str = Field(default="", description="Agent's reasoning")
    action: str = Field(default="", description="Tool name or 'final_answer'")
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments or final answer content"
    )
    is_final_answer: bool = Field(
        default=False,
        description="True if action == 'final_answer'"
    )
    raw_response: str = Field(default="", description="Original LLM output")
    parse_error: str = Field(default="", description="Error details if parsing failed")

    @property
    def is_valid(self) -> bool:
        """A response is valid if it has a thought and either an action or final answer."""
        return bool(self.thought) and bool(self.action)


# ===================================================================
# Parser Functions
# ===================================================================

def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences that LLMs love to add.

    Handles: ```json {...} ```, ``` {...} ```, and variants.
    """
    # Remove ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object from a string by matching braces.

    This handles cases where the LLM adds text before/after the JSON.
    Uses brace counting to handle nested objects correctly.
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _fix_common_json_issues(text: str) -> str:
    """
    Attempt to fix common JSON formatting issues.

    These are heuristic fixes — not guaranteed to work, but
    they recover a surprising number of almost-valid responses.
    """
    # Replace single quotes with double quotes (outside of values)
    # This is a naive fix — it can break strings containing apostrophes
    # but it's better than crashing
    # Only do this if there are no double quotes at all
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix unquoted keys (simple cases)
    text = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1 "\2":', text)

    return text


def parse_react_response(raw_response: str) -> ParsedResponse:
    """
    Parse an LLM response into a structured ParsedResponse.

    Strategy (in order of strictness):
    1. Try direct JSON parse
    2. Strip code fences and try again
    3. Extract JSON object by brace matching
    4. Apply common fixes and try again
    5. Give up and return error

    Args:
        raw_response: Raw string from the LLM.

    Returns:
        ParsedResponse — always valid, may contain parse_error.
    """
    if not raw_response or not raw_response.strip():
        return ParsedResponse(
            raw_response=raw_response or "",
            parse_error="Empty response from LLM",
        )

    cleaned = raw_response.strip()

    # --- Strategy 1: Direct JSON parse ---
    parsed_dict = _try_json_parse(cleaned)

    # --- Strategy 2: Strip code fences ---
    if parsed_dict is None:
        stripped = _strip_code_fences(cleaned)
        parsed_dict = _try_json_parse(stripped)

    # --- Strategy 3: Extract JSON object ---
    if parsed_dict is None:
        extracted = _extract_json_object(cleaned)
        if extracted:
            parsed_dict = _try_json_parse(extracted)

    # --- Strategy 4: Fix common issues ---
    if parsed_dict is None:
        fixed = _fix_common_json_issues(cleaned)
        parsed_dict = _try_json_parse(fixed)

        # Also try after stripping fences + fixing
        if parsed_dict is None:
            fixed_stripped = _fix_common_json_issues(_strip_code_fences(cleaned))
            parsed_dict = _try_json_parse(fixed_stripped)

    # --- Strategy 5: Give up ---
    if parsed_dict is None:
        logger.warning(f"Failed to parse LLM response: {cleaned[:200]}...")
        return ParsedResponse(
            raw_response=raw_response,
            parse_error=f"Could not extract valid JSON from response. Raw: {cleaned[:300]}",
        )

    # --- Validate and build response ---
    return _build_parsed_response(parsed_dict, raw_response)


def _try_json_parse(text: str) -> Optional[dict]:
    """Attempt to parse text as JSON. Returns dict or None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _build_parsed_response(data: dict, raw_response: str) -> ParsedResponse:
    """
    Build a ParsedResponse from a parsed dict, with validation.

    Handles edge cases:
    - action_input as string instead of dict
    - missing thought field
    - missing action field
    """
    thought = str(data.get("thought", ""))
    action = str(data.get("action", ""))
    action_input = data.get("action_input", {})

    # Handle action_input being a string (common LLM mistake)
    if isinstance(action_input, str):
        # Try to parse it as JSON
        try:
            action_input = json.loads(action_input)
        except (json.JSONDecodeError, ValueError):
            # If the action is final_answer, wrap the string
            if action == "final_answer":
                action_input = {"answer": action_input}
            else:
                action_input = {"raw_input": action_input}

    if not isinstance(action_input, dict):
        action_input = {"value": action_input}

    # Check for final_answer
    is_final = action.lower().strip() == "final_answer"

    # Also check if there's a final_answer field directly
    if not is_final and "final_answer" in data and data["final_answer"]:
        is_final = True
        action = "final_answer"
        answer = data["final_answer"]
        if isinstance(answer, str):
            action_input = {"answer": answer}
        elif isinstance(answer, dict):
            action_input = answer

    # Build validation error if fields are missing
    parse_error = ""
    if not thought:
        parse_error = "Missing 'thought' field in response. "
    if not action and not is_final:
        parse_error += "Missing 'action' field in response."

    return ParsedResponse(
        thought=thought,
        action=action,
        action_input=action_input,
        is_final_answer=is_final,
        raw_response=raw_response,
        parse_error=parse_error.strip(),
    )
