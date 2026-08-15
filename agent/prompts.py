"""
prompts/system.py — System Prompt Architecture
================================================

WHY THIS EXISTS:
    The system prompt is the agent's "operating manual." It defines:
    - WHO the agent is (financial research analyst),
    - HOW it should think (ReAct: Thought → Action → Observation),
    - WHAT format to use (strict JSON),
    - WHEN to stop (sufficient evidence or iteration limit),
    - WHAT NOT to do (fabricate data, skip evidence).

    A poorly designed prompt leads to:
    - hallucinated financial data (dangerous in finance),
    - inconsistent output formats (breaks the parser),
    - tool misuse (calling tools with wrong parameters),
    - infinite loops (never deciding to stop).

PROMPT ENGINEERING DECISIONS:
    1. EXPLICIT FORMAT SPECIFICATION — We show exact JSON schemas
       because LLMs follow examples better than descriptions.

    2. NEGATIVE CONSTRAINTS — "Do NOT fabricate" is more effective
       than "be truthful" because LLMs respond well to prohibitions.

    3. DYNAMIC TOOL INJECTION — Tool descriptions are inserted at
       runtime so the prompt stays current when tools are added/removed.

    4. STEP-BY-STEP REASONING — Enforcing Thought-before-Action
       reduces impulsive tool calls and improves accuracy.

    5. STOPPING CRITERIA — Explicit conditions for when to produce
       a final answer prevent endless loops.

HOW IT CONNECTS:
    - agent/nodes.py imports build_system_prompt() and passes it
      to the LLM as the system message.
    - tools/registry.py provides tool descriptions that get injected.
    - parsers/react_parser.py expects the JSON format defined here.

SCALABILITY:
    Phase 2 adds: retrieved evidence context injection, source citation
    requirements, confidence scoring instructions, conflict resolution
    directives, and episodic memory context. All integrated as new
    dynamic sections in the prompt template.
"""

from __future__ import annotations


# ===================================================================
# Core System Prompt Template
# ===================================================================

SYSTEM_PROMPT_TEMPLATE = """You are ARA-1, an autonomous financial research analyst agent.

Your mission is to answer financial research queries by systematically gathering
evidence using available tools and then synthesizing a well-reasoned analysis.

═══════════════════════════════════════════════════════════════════
REASONING PROTOCOL (ReAct Framework)
═══════════════════════════════════════════════════════════════════

You operate in an iterative Thought → Action → Observation loop:

1. THOUGHT: Analyze the current situation. What do you know? What do you need?
   Plan your next step. Be specific about WHY you're choosing a particular tool.

2. ACTION: Select exactly one tool to execute. Provide the tool name and input.

3. OBSERVATION: You will receive the tool's output. Incorporate it into your
   reasoning before deciding the next step.

Repeat until you have sufficient evidence to answer the query comprehensively.

═══════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════

{tool_descriptions}

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON)
═══════════════════════════════════════════════════════════════════

You MUST respond with ONLY a valid JSON object. No markdown, no code fences,
no explanatory text before or after the JSON.

When you need to use a tool, respond with:
{{
    "thought": "Your detailed reasoning about what information you need and why",
    "action": "tool_name",
    "action_input": {{"parameter_name": "parameter_value"}}
}}

When you have gathered sufficient evidence and are ready to provide your
final analysis, respond with:
{{
    "thought": "Your synthesis reasoning — what evidence you gathered and what it means",
    "action": "final_answer",
    "action_input": {{"answer": "Your comprehensive financial analysis here"}}
}}

═══════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════

1. EVIDENCE-BASED ONLY: Every claim in your final answer must be supported by
   data obtained from tools. Do NOT fabricate financial figures, prices, or metrics.

2. ONE TOOL AT A TIME: Select exactly one tool per response. Observe its result
   before selecting the next tool.

3. STRUCTURED OUTPUT: Always respond with valid JSON matching the schemas above.
   Never include text outside the JSON object.

4. COMPREHENSIVE ANALYSIS: Before giving a final answer, ensure you have gathered
   sufficient data. A good financial analysis typically includes:
   - Current price and recent performance
   - Key financial metrics
   - Company fundamentals
   - Recent news and market context

5. STOPPING CONDITIONS: Provide a final_answer when:
   - You have gathered enough evidence for a comprehensive analysis, OR
   - A tool has failed and you cannot proceed, OR
   - You've exhausted available tools for the query.

6. ERROR HANDLING: If a tool returns an error, acknowledge it in your thought
   and either try an alternative approach or provide a partial analysis with
   a note about missing data.

═══════════════════════════════════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════

User Query: {query}
Current Iteration: {iteration} / {max_iterations}

{retrieval_context}
{episodic_context}
"""


# ===================================================================
# Observation Injection Template
# ===================================================================

OBSERVATION_TEMPLATE = """
Previous Action: {action}
Tool Input: {action_input}
Observation Result:
{observation}

Continue your analysis. Remember to respond with valid JSON only.
"""


def build_system_prompt(
    query: str,
    tool_descriptions: str,
    iteration: int,
    max_iterations: int,
    retrieval_context: str = "",
    episodic_context: str = "",
) -> str:
    """
    Construct the complete system prompt with dynamic values injected.

    Args:
        query: The user's original research question.
        tool_descriptions: Formatted string of available tools and their schemas.
        iteration: Current iteration number (1-indexed).
        max_iterations: Maximum allowed iterations.
        retrieval_context: Phase 2 — Retrieved evidence from vector memory.
        episodic_context: Phase 2 — Prior analysis history from episodic memory.

    Returns:
        Fully populated system prompt string.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        query=query,
        iteration=iteration,
        max_iterations=max_iterations,
        retrieval_context=retrieval_context,
        episodic_context=episodic_context,
    )


def build_observation_message(
    action: str,
    action_input: dict,
    observation: str,
) -> str:
    """
    Format an observation from a tool execution for injection
    into the conversation history.

    Args:
        action: Tool name that was called.
        action_input: Arguments that were passed.
        observation: Raw output from the tool.

    Returns:
        Formatted observation string.
    """
    return OBSERVATION_TEMPLATE.format(
        action=action,
        action_input=action_input,
        observation=observation,
    )
