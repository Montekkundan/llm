from __future__ import annotations

from picollm.common.conversation import STRUCTURED_ROLE_TOKENS


def _extract_message_text(item: object) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    role = str(item.get("role", "")).strip().lower()
    content = item.get("content", "")
    content_str = str(content).strip()
    if role not in {"system", "user", "assistant"} or not content_str:
        return None
    return role, content_str


def normalize_text(value: object, alternating_chat_roles: bool) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            rendered: list[str] = []
            for item in value:
                extracted = _extract_message_text(item)
                if extracted is None:
                    continue
                role, content = extracted
                start_token, end_token = STRUCTURED_ROLE_TOKENS[role]
                rendered.append(f"{start_token}{content}{end_token}")
            return "\n".join(rendered)

        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return ""
        if alternating_chat_roles:
            rendered = []
            for index, part in enumerate(parts):
                role = "user" if index % 2 == 0 else "assistant"
                start_token, end_token = STRUCTURED_ROLE_TOKENS[role]
                rendered.append(f"{start_token}{part}{end_token}")
            return "\n".join(rendered)
        return "\n".join(parts)
    return str(value).strip()
