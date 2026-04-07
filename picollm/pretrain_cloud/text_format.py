from __future__ import annotations


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "system": "system",
}


def _normalize_role(role: object) -> str:
    normalized = str(role).strip().lower()
    return ROLE_ALIASES.get(normalized, normalized or "user")


def _extract_message_text(item: object) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    role = item.get("role", item.get("from", item.get("speaker", "user")))
    content = item.get("content", item.get("value", item.get("text", "")))
    content_str = str(content).strip()
    if not content_str:
        return None
    return _normalize_role(role), content_str


def normalize_text(value: object, alternating_chat_roles: bool) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            rendered: list[str] = []
            for item in value:
                extracted = _extract_message_text(item)
                if extracted is None:
                    continue
                role, content = extracted
                rendered.append(f"<|{role}|> {content}")
            return "\n".join(rendered)

        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return ""
        if alternating_chat_roles:
            rendered = []
            for index, part in enumerate(parts):
                role = "<|user|>" if index % 2 == 0 else "<|assistant|>"
                rendered.append(f"{role} {part}")
            return "\n".join(rendered)
        return "\n".join(parts)
    return str(value).strip()
