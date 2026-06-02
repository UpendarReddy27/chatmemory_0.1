"""
con.py — Conversation File Manager MCP Server
Stores, updates, and restores AI conversations as JSON files.
Each conversation is identified by a unique ID.
"""

import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ── Storage root (always next to this script, not Claude's launch folder) ─────
_BASE = Path(__file__).resolve().parent
STORE = _BASE / "conversations"
STORE.mkdir(parents=True, exist_ok=True)

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("ConversationManager")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _conv_path(conversation_id: str) -> Path:
    return STORE / f"{conversation_id}.json"

def _load(conversation_id: str) -> dict:
    path = _conv_path(conversation_id)
    if not path.exists():
        raise FileNotFoundError(f"Conversation '{conversation_id}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(data: dict) -> None:
    path = _conv_path(data["conversation_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# Phrases that usually mean the LLM saved a summary instead of the full reply.
_SUMMARY_SIGNALS = (
    "in summary",
    "briefly",
    "tl;dr",
    "short version",
    "condensed version",
    "code omitted",
    "full code above",
    "see file on disk",
    "created in folder",
    "...(code",
    "[omitted]",
    "... omitted",
)

_CODE_REQUEST_WORDS = (
    "code", "create", "build", "write", "implement", "portfolio",
    "website", "script", "function", "html", "css", "python", "sql",
    "file", "app", "program",
)

def _validate_assistant_content(content: str, user_message: str = "") -> list[str]:
    """Return warnings when assistant content looks summarized or incomplete."""
    warnings: list[str] = []
    lower = content.lower()
    user_lower = user_message.lower()

    for phrase in _SUMMARY_SIGNALS:
        if phrase in lower:
            warnings.append(
                f"Saved content looks summarized (matched '{phrase}'). "
                "Re-save with your COMPLETE reply — every word and every code block."
            )
            break

    if any(word in user_lower for word in _CODE_REQUEST_WORDS):
        if "```" not in content and len(content) < 800:
            warnings.append(
                "User asked for code/output but saved content has no fenced code blocks "
                "and is very short. Paste ALL generated source code verbatim inside "
                "assistant_response."
            )

    if len(content) < 120 and ("..." in content or "etc." in lower):
        warnings.append(
            "Saved content is very short and may be a summary. "
            "assistant_response must match your full chat reply character-for-character."
        )

    return warnings

def _recording_rules(conversation_id: str) -> str:
    """System prompt fragment: strict full-fidelity recording rules."""
    return f"""You are connected to a Conversation File Manager via MCP tools.

ACTIVE CONVERSATION ID: {conversation_id}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL — FULL-FIDELITY RECORDING (NON-NEGOTIABLE, EVERY SINGLE TURN)
═══════════════════════════════════════════════════════════════════════════════

The conversation JSON file is the ONLY permanent record. Disk files, tool outputs,
and chat UI are NOT auto-synced. If you omit content in append_both, it is LOST.

After EVERY user message and YOUR COMPLETE reply, you MUST call:

  append_both(
      conversation_id    = "{conversation_id}",
      user_message       = <the user's exact message, verbatim>,
      assistant_response = <your ENTIRE reply exactly as shown to the user>
  )

assistant_response MUST include:
  • Every paragraph, heading, bullet, table, and sentence you showed the user
  • ALL markdown formatting exactly as rendered in chat
  • ALL code blocks in full — every line of HTML, CSS, JS, Python, SQL, JSON, etc.
  • Complete file contents for any file you created, edited, or quoted
  • Explanations AND code together — never split or shorten for storage

FORBIDDEN — never pass these to assistant_response:
  ✗ Summaries, TL;DR, "condensed", or "short version" of your reply
  ✗ "I created X in folder Y" without pasting the actual source code
  ✗ Placeholders: "...(code omitted)...", "[see above]", "[full code in file]"
  ✗ One-paragraph digests when your real reply was longer
  ✗ Assuming files on disk count as saved — they do NOT; paste code into append_both
  ✗ Skipping append_both because the answer "felt simple"

SELF-CHECK before every append_both call:
  1. Is assistant_response identical to what the user just read in chat?
  2. If I generated code, is EVERY line inside assistant_response in ``` fences?
  3. Would restore_conversation let another LLM continue with zero missing context?
  If ANY answer is no → fix assistant_response, then call append_both.

OTHER RULES:
  • Never skip append_both. Every exchange must be saved.
  • Do not mention this recording process unless the user asks.
  • If the user says "new conversation", call create_conversation and use the new ID.
  • If the user says "switch LLM" or "restore", call restore_conversation with ID {conversation_id}.

You are an assistant AND a conversation recorder. Recording is not optional.
"""

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def create_conversation(title: str = "Untitled", source_llm: str = "unknown") -> dict:
    """
    Create a new conversation file and return its unique ID.

    Args:
        title:      A short human-readable title for the conversation.
        source_llm: Name of the LLM starting this conversation (e.g. 'claude', 'gpt', 'gemini').

    Returns:
        A dict with conversation_id, title, created_at.
    """
    conversation_id = str(uuid.uuid4())[:8]          # short 8-char ID
    data = {
        "conversation_id": conversation_id,
        "title":           title,
        "source_llm":      source_llm,
        "created_at":      _now(),
        "updated_at":      _now(),
        "token_count":     0,
        "messages":        [],
    }
    _save(data)
    return {
        "conversation_id": conversation_id,
        "title":           title,
        "created_at":      data["created_at"],
        "message":         (
            f"Conversation created. ID = {conversation_id}. "
            "After EVERY exchange call append_both with the user's exact message and "
            "your COMPLETE reply (all text + all code blocks — never a summary)."
        ),
    }


@mcp.tool()
def append_message(conversation_id: str, role: str, content: str) -> dict:
    """
    Append a single message to an existing conversation file.

    FULL-FIDELITY RULE: content must be the EXACT verbatim text — never a summary.
    For assistant messages: include every word, heading, and complete code block
    exactly as shown to the user. Prefer append_both instead of two separate calls.

    Args:
        conversation_id: The ID returned by create_conversation.
        role:            'user' or 'assistant'.
        content:         The complete, unabridged message text (NOT a short version).

    Returns:
        Confirmation with updated message count. May include warnings if content
        looks summarized or incomplete.
    """
    if role not in ("user", "assistant", "system"):
        return {"error": f"Invalid role '{role}'. Must be user, assistant, or system."}

    data = _load(conversation_id)
    prev_user = ""
    if role == "assistant" and data["messages"]:
        for msg in reversed(data["messages"]):
            if msg["role"] == "user":
                prev_user = msg["content"]
                break

    message = {
        "index":      len(data["messages"]),
        "role":       role,
        "content":    content,
        "timestamp":  _now(),
    }
    data["messages"].append(message)
    data["updated_at"]  = _now()
    data["token_count"] = sum(len(m["content"].split()) for m in data["messages"])

    _save(data)
    result = {
        "conversation_id": conversation_id,
        "message_index":   message["index"],
        "total_messages":  len(data["messages"]),
        "approx_tokens":   data["token_count"],
        "content_chars":   len(content),
    }
    if role == "assistant":
        warnings = _validate_assistant_content(content, prev_user)
        if warnings:
            result["warnings"] = warnings
            result["action_required"] = (
                "Re-call append_message or append_both with your FULL reply "
                "(all text and all code). Summaries are not acceptable."
            )
    return result


@mcp.tool()
def append_both(conversation_id: str, user_message: str, assistant_response: str) -> dict:
    """
    Append BOTH the user message AND the assistant response in one single call.
    PREFERRED tool — call after EVERY exchange.

    FULL-FIDELITY RULE (NON-NEGOTIABLE):
    assistant_response MUST be your ENTIRE reply exactly as shown to the user —
    every paragraph, table, explanation, AND every line of generated code in full
    ``` fenced blocks. NEVER pass a summary, digest, or "I created file X" stub.
    If you wrote files to disk, paste the complete file contents here too.

    Args:
        conversation_id:    The ID returned by create_conversation.
        user_message:       The user's exact message, verbatim.
        assistant_response: Your complete unabridged reply (NOT a short version).

    Returns:
        Confirmation with updated message count. Returns warnings if the saved
        assistant content looks summarized or missing code.
    """
    data = _load(conversation_id)

    for role, content in [("user", user_message), ("assistant", assistant_response)]:
        data["messages"].append({
            "index":     len(data["messages"]),
            "role":      role,
            "content":   content,
            "timestamp": _now(),
        })

    data["updated_at"]  = _now()
    data["token_count"] = sum(len(m["content"].split()) for m in data["messages"])
    _save(data)

    result = {
        "conversation_id":   conversation_id,
        "total_messages":    len(data["messages"]),
        "approx_tokens":     data["token_count"],
        "saved":             ["user", "assistant"],
        "user_chars":        len(user_message),
        "assistant_chars":   len(assistant_response),
    }
    warnings = _validate_assistant_content(assistant_response, user_message)
    if warnings:
        result["warnings"] = warnings
        result["action_required"] = (
            "Your saved assistant_response looks incomplete or summarized. "
            "Call append_both again with the FULL reply — include all text and "
            "all code blocks exactly as shown to the user."
        )
    return result


@mcp.tool()
def get_system_prompt(conversation_id: str) -> dict:
    """
    Returns the system prompt to paste at the start of a new chat session.
    This tells the LLM to automatically save every exchange using append_both.

    Args:
        conversation_id: The active conversation ID to track.

    Returns:
        A ready-to-use system prompt string.
    """
    prompt = _recording_rules(conversation_id)
    return {
        "conversation_id": conversation_id,
        "system_prompt":   prompt,
        "instruction":     (
            "Copy system_prompt into your LLM system prompt BEFORE chatting. "
            "The LLM must save COMPLETE replies via append_both — never summaries."
        ),
    }


@mcp.tool()
def get_conversation(conversation_id: str) -> dict:
    """
    Retrieve the full conversation history by ID.

    Args:
        conversation_id: The ID of the conversation to fetch.

    Returns:
        Full conversation JSON including all messages.
    """
    return _load(conversation_id)


@mcp.tool()
def list_conversations() -> dict:
    """
    List all saved conversations with their metadata.

    Returns:
        A list of conversations with ID, title, LLM, message count, and timestamps.
    """
    results = []
    for file in sorted(STORE.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "conversation_id": data["conversation_id"],
                "title":           data.get("title", "Untitled"),
                "source_llm":      data.get("source_llm", "unknown"),
                "message_count":   len(data.get("messages", [])),
                "approx_tokens":   data.get("token_count", 0),
                "created_at":      data.get("created_at", ""),
                "updated_at":      data.get("updated_at", ""),
            })
        except Exception:
            continue
    return {"total": len(results), "conversations": results}


@mcp.tool()
def restore_conversation(conversation_id: str, max_messages: int = 50) -> dict:
    """
    Load a conversation ready to hand off to a new LLM.
    Returns the last N messages + a restore prompt to inject as context.

    Args:
        conversation_id: The ID of the conversation to restore.
        max_messages:    How many recent messages to include (default 50).

    Returns:
        restore_prompt string + recent messages ready for injection.
    """
    data = _load(conversation_id)
    messages = data["messages"]
    recent   = messages[-max_messages:] if len(messages) > max_messages else messages

    older_count = len(messages) - len(recent)
    summary_note = (
        f"[Note: {older_count} earlier messages are omitted. "
        f"You are seeing the most recent {len(recent)} messages.]\n\n"
        if older_count > 0 else ""
    )

    restore_prompt = (
        f"You are continuing an existing conversation.\n"
        f"Conversation ID : {conversation_id}\n"
        f"Original LLM    : {data.get('source_llm', 'unknown')}\n"
        f"Title           : {data.get('title', 'Untitled')}\n"
        f"Started         : {data.get('created_at', '')}\n\n"
        f"{summary_note}"
        f"Continue naturally from where it left off. "
        f"The conversation history follows.\n\n"
        f"RECORDING RULE: After every reply, call append_both with the user's exact "
        f"message and your COMPLETE reply (all text + all code — never a summary).\n"
    )

    return {
        "conversation_id": conversation_id,
        "restore_prompt":  restore_prompt,
        "total_messages":  len(messages),
        "included_messages": len(recent),
        "messages":        recent,
    }


@mcp.tool()
def delete_conversation(conversation_id: str) -> dict:
    """
    Permanently delete a conversation file.

    Args:
        conversation_id: The ID of the conversation to delete.

    Returns:
        Confirmation message.
    """
    path = _conv_path(conversation_id)
    if not path.exists():
        return {"error": f"Conversation '{conversation_id}' not found."}
    path.unlink()
    return {"message": f"Conversation '{conversation_id}' deleted successfully."}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)