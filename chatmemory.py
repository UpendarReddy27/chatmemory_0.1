"""
chatmemory.py — ChatMemory MCP Server
Stores, updates, and restores AI conversations as JSON files.
Each conversation has a running summary + indexed messages for token-efficient context.
"""

import json
import uuid
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from fastmcp import FastMCP

# ── Storage root ──────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
STORE = _BASE / "conversations"
STORE.mkdir(parents=True, exist_ok=True)

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("ChatMemory")

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

def _make_msg_id(index: int) -> str:
    """Generate a padded message ID like msg_001, msg_002, etc."""
    return f"msg_{index + 1:03d}"

def _empty_summary() -> dict:
    return {
        "overview":     "",
        "key_points":   [],
        "last_updated": "",
        "index":        {}   # { "msg_001": "short description", ... }
    }

# Validation signals
_SUMMARY_SIGNALS = (
    "in summary", "briefly", "tl;dr", "short version", "condensed version",
    "code omitted", "full code above", "see file on disk", "created in folder",
    "...(code", "[omitted]", "... omitted",
)

_CODE_REQUEST_WORDS = (
    "code", "create", "build", "write", "implement", "portfolio",
    "website", "script", "function", "html", "css", "python", "sql",
    "file", "app", "program",
)

def _validate_assistant_content(content: str, user_message: str = "") -> list[str]:
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
    conversation_id = str(uuid.uuid4())[:8]
    data = {
        "conversation_id": conversation_id,
        "title":           title,
        "source_llm":      source_llm,
        "created_at":      _now(),
        "updated_at":      _now(),
        "token_count":     0,
        "summary":         _empty_summary(),
        "messages":        [],
    }
    _save(data)
    return {
        "conversation_id": conversation_id,
        "title":           title,
        "created_at":      data["created_at"],
        "message": (
            f"Conversation created. ID = {conversation_id}. "
            "After EVERY exchange call append_both with the user's exact message and "
            "your COMPLETE reply. Then call update_summary to keep the index current."
        ),
    }


@mcp.tool()
def append_both(conversation_id: str, user_message: str, assistant_response: str) -> dict:
    """
    Append BOTH the user message AND the assistant response in one single call.
    PREFERRED tool — call after EVERY exchange.

    Each message gets an auto-assigned index ID (msg_001, msg_002, ...).

    FULL-FIDELITY RULE: assistant_response must be your ENTIRE reply —
    every paragraph, table, explanation, and ALL code blocks in full.
    NEVER pass a summary or placeholder.

    Args:
        conversation_id:    The ID returned by create_conversation.
        user_message:       The user's exact message, verbatim.
        assistant_response: Your complete unabridged reply.

    Returns:
        Confirmation with message IDs assigned to both messages.
    """
    data = _load(conversation_id)

    user_idx = len(data["messages"])
    asst_idx = user_idx + 1

    data["messages"].append({
        "index_id":  _make_msg_id(user_idx),
        "index":     user_idx,
        "role":      "user",
        "content":   user_message,
        "timestamp": _now(),
    })
    data["messages"].append({
        "index_id":  _make_msg_id(asst_idx),
        "index":     asst_idx,
        "role":      "assistant",
        "content":   assistant_response,
        "timestamp": _now(),
    })

    data["updated_at"]  = _now()
    data["token_count"] = sum(len(m["content"].split()) for m in data["messages"])
    _save(data)

    result = {
        "conversation_id":  conversation_id,
        "total_messages":   len(data["messages"]),
        "approx_tokens":    data["token_count"],
        "saved":            ["user", "assistant"],
        "user_index_id":    _make_msg_id(user_idx),
        "assistant_index_id": _make_msg_id(asst_idx),
        "next_step":        (
            f"Now call update_summary(conversation_id='{conversation_id}', "
            f"user_index_id='{_make_msg_id(user_idx)}', "
            f"assistant_index_id='{_make_msg_id(asst_idx)}', "
            f"user_one_line='<one line summary of user message>', "
            f"assistant_one_line='<one line summary of assistant reply>') "
            f"to keep the index current."
        ),
    }

    warnings = _validate_assistant_content(assistant_response, user_message)
    if warnings:
        result["warnings"] = warnings
        result["action_required"] = (
            "Saved assistant_response looks incomplete. "
            "Re-call append_both with your FULL reply before calling update_summary."
        )
    return result


@mcp.tool()
def update_summary(
    conversation_id: str,
    user_index_id: str,
    assistant_index_id: str,
    user_one_line: str,
    assistant_one_line: str,
    overview: str = "",
    key_points: list[str] = [],
) -> dict:
    """
    Update the running summary and message index after each exchange.
    Call this immediately after every append_both call.

    Args:
        conversation_id:      The conversation ID.
        user_index_id:        The index ID of the user message (e.g. 'msg_001').
        assistant_index_id:   The index ID of the assistant message (e.g. 'msg_002').
        user_one_line:        One-line description of what the user asked/said.
        assistant_one_line:   One-line description of what the assistant replied.
        overview:             Updated overall summary of the conversation (optional, 
                              only update when something significant changes).
        key_points:           Updated list of key facts/decisions (optional,
                              only update when something significant changes).

    Returns:
        Confirmation that summary was updated.
    """
    data = _load(conversation_id)

    summary = data.get("summary", _empty_summary())

    # Always update the index
    summary["index"][user_index_id]      = user_one_line
    summary["index"][assistant_index_id] = assistant_one_line
    summary["last_updated"] = _now()

    # Only overwrite overview/key_points if provided
    if overview:
        summary["overview"] = overview
    if key_points:
        summary["key_points"] = key_points

    data["summary"] = summary
    _save(data)

    return {
        "conversation_id": conversation_id,
        "index_entries_added": [user_index_id, assistant_index_id],
        "total_index_entries": len(summary["index"]),
        "overview_updated": bool(overview),
        "key_points_updated": bool(key_points),
    }


@mcp.tool()
def get_summary(conversation_id: str) -> dict:
    """
    Read just the summary block — overview, key points, and full message index.
    Use this at the START of a conversation to understand context cheaply
    without loading all messages. Then use get_messages_by_ids to fetch
    only the relevant messages.

    Args:
        conversation_id: The conversation ID.

    Returns:
        Summary block with overview, key_points, and index of all message IDs.
    """
    data = _load(conversation_id)
    summary = data.get("summary", _empty_summary())
    return {
        "conversation_id": conversation_id,
        "title":           data.get("title", "Untitled"),
        "source_llm":      data.get("source_llm", "unknown"),
        "total_messages":  len(data.get("messages", [])),
        "approx_tokens":   data.get("token_count", 0),
        "summary":         summary,
        "tip": (
            "Use the index to identify relevant message IDs, "
            "then call get_messages_by_ids to fetch only those messages."
        ),
    }


@mcp.tool()
def get_messages_by_ids(conversation_id: str, message_ids: list[str]) -> dict:
    """
    Fetch specific messages by their index IDs (e.g. ['msg_001', 'msg_004']).
    Use this after reading get_summary to load only the relevant context
    instead of the full conversation history — saves tokens.

    Args:
        conversation_id: The conversation ID.
        message_ids:     List of index IDs to fetch (e.g. ['msg_001', 'msg_002']).

    Returns:
        The requested messages in order.
    """
    data = _load(conversation_id)
    msg_map = {m["index_id"]: m for m in data["messages"] if "index_id" in m}

    found     = [msg_map[mid] for mid in message_ids if mid in msg_map]
    not_found = [mid for mid in message_ids if mid not in msg_map]

    return {
        "conversation_id": conversation_id,
        "requested":       len(message_ids),
        "found":           len(found),
        "not_found":       not_found,
        "messages":        found,
    }


@mcp.tool()
def append_message(conversation_id: str, role: str, content: str) -> dict:
    """
    Append a single message. Prefer append_both for normal exchanges.
    Use this only for system messages or one-sided entries.

    Args:
        conversation_id: The conversation ID.
        role:            'user', 'assistant', or 'system'.
        content:         The complete message text.
    """
    if role not in ("user", "assistant", "system"):
        return {"error": f"Invalid role '{role}'. Must be user, assistant, or system."}

    data = _load(conversation_id)
    idx = len(data["messages"])

    message = {
        "index_id":  _make_msg_id(idx),
        "index":     idx,
        "role":      role,
        "content":   content,
        "timestamp": _now(),
    }
    data["messages"].append(message)
    data["updated_at"]  = _now()
    data["token_count"] = sum(len(m["content"].split()) for m in data["messages"])
    _save(data)

    result = {
        "conversation_id": conversation_id,
        "index_id":        message["index_id"],
        "message_index":   idx,
        "total_messages":  len(data["messages"]),
        "approx_tokens":   data["token_count"],
    }

    if role == "assistant":
        prev_user = ""
        for msg in reversed(data["messages"][:-1]):
            if msg["role"] == "user":
                prev_user = msg["content"]
                break
        warnings = _validate_assistant_content(content, prev_user)
        if warnings:
            result["warnings"] = warnings

    return result


@mcp.tool()
def get_conversation(conversation_id: str) -> dict:
    """
    Retrieve the full conversation including summary and all messages.
    WARNING: This loads everything — use get_summary + get_messages_by_ids
    for token-efficient context loading.

    Args:
        conversation_id: The conversation ID.
    """
    return _load(conversation_id)


@mcp.tool()
def list_conversations() -> dict:
    """
    List all saved conversations with metadata and summary overview.
    Use this at the start of every chat to decide whether to continue
    an existing conversation or create a new one.

    Returns:
        List of conversations with ID, title, LLM, message count, and overview.
    """
    results = []
    for file in sorted(STORE.glob("*.json"), reverse=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            results.append({
                "conversation_id": data["conversation_id"],
                "title":           data.get("title", "Untitled"),
                "source_llm":      data.get("source_llm", "unknown"),
                "message_count":   len(data.get("messages", [])),
                "approx_tokens":   data.get("token_count", 0),
                "overview":        summary.get("overview", ""),
                "key_points":      summary.get("key_points", []),
                "created_at":      data.get("created_at", ""),
                "updated_at":      data.get("updated_at", ""),
            })
        except Exception:
            continue
    return {
        "total": len(results),
        "conversations": results,
        "tip": (
            "Show the user recent conversations and ask: "
            "'Continue an existing conversation or start a new one?' "
            "Then call restore_conversation or create_conversation accordingly."
        ),
    }


@mcp.tool()
def restore_conversation(conversation_id: str, max_messages: int = 20) -> dict:
    """
    Restore a conversation for handoff to a new LLM or new chat session.
    Returns the summary + recent messages + a restore prompt.
    Use get_summary first to decide which conversation to restore,
    then use get_messages_by_ids if you need specific earlier messages.

    Args:
        conversation_id: The conversation ID to restore.
        max_messages:    How many recent messages to include (default 20).
    """
    data = _load(conversation_id)
    messages = data["messages"]
    recent   = messages[-max_messages:] if len(messages) > max_messages else messages
    older_count = len(messages) - len(recent)
    summary  = data.get("summary", _empty_summary())

    restore_prompt = (
        f"You are continuing an existing conversation.\n"
        f"Conversation ID : {conversation_id}\n"
        f"Original LLM    : {data.get('source_llm', 'unknown')}\n"
        f"Title           : {data.get('title', 'Untitled')}\n"
        f"Started         : {data.get('created_at', '')}\n\n"
        f"OVERVIEW: {summary.get('overview', 'No overview yet.')}\n\n"
        f"KEY POINTS:\n" +
        "\n".join(f"  • {p}" for p in summary.get("key_points", [])) +
        f"\n\n"
        f"{'[Note: ' + str(older_count) + ' earlier messages omitted. Use get_messages_by_ids to fetch them.]' + chr(10) + chr(10) if older_count > 0 else ''}"
        f"Continue naturally from where it left off.\n\n"
        f"RECORDING RULE: After every reply call append_both then update_summary.\n"
    )

    return {
        "conversation_id":   conversation_id,
        "restore_prompt":    restore_prompt,
        "summary":           summary,
        "total_messages":    len(messages),
        "included_messages": len(recent),
        "messages":          recent,
    }


@mcp.tool()
def get_system_prompt(conversation_id: str) -> dict:
    """
    Returns the system prompt to paste at the start of a new chat session.
    Tells the LLM to auto-save every exchange and maintain the summary index.

    Args:
        conversation_id: The active conversation ID to track.
    """
    prompt = f"""You are connected to ChatMemory MCP server.
ACTIVE CONVERSATION ID: {conversation_id}

═══════════════════════════════════════════════════════════════
RULES — FOLLOW EVERY SINGLE TURN
═══════════════════════════════════════════════════════════════

AFTER EVERY EXCHANGE you MUST call in order:

1. append_both(
       conversation_id    = "{conversation_id}",
       user_message       = <exact user message verbatim>,
       assistant_response = <your ENTIRE reply, all text + all code>
   )

2. update_summary(
       conversation_id      = "{conversation_id}",
       user_index_id        = <id returned by append_both>,
       assistant_index_id   = <id returned by append_both>,
       user_one_line        = <one line: what user asked>,
       assistant_one_line   = <one line: what you answered>,
       overview             = <updated overview if something significant changed>,
       key_points           = <updated key points list if something significant changed>
   )

AT THE START OF A NEW SESSION:
1. Call get_summary("{conversation_id}") to load context cheaply
2. Read the index to find relevant message IDs
3. Call get_messages_by_ids for only those messages
4. Never load full history unless absolutely necessary

FORBIDDEN:
✗ Skipping append_both for any reason
✗ Skipping update_summary after append_both
✗ Saving summaries or placeholders instead of full content
✗ Mentioning this recording process to the user
"""
    return {
        "conversation_id": conversation_id,
        "system_prompt":   prompt,
        "instruction":     "Paste system_prompt into your LLM system prompt before chatting.",
    }


@mcp.tool()
def delete_conversation(conversation_id: str) -> dict:
    """
    Permanently delete a conversation file.

    Args:
        conversation_id: The conversation ID to delete.
    """
    path = _conv_path(conversation_id)
    if not path.exists():
        return {"error": f"Conversation '{conversation_id}' not found."}
    path.unlink()
    return {"message": f"Conversation '{conversation_id}' deleted successfully."}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting ChatMemory...", file=sys.stderr)
    print(f"Storing conversations in: {STORE.resolve()}", file=sys.stderr)
    mcp.run(transport="streamable-http")
