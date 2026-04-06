from __future__ import annotations

import requests


if __name__ == '__main__':
    base = 'http://127.0.0.1:8000'
    print('health:', requests.get(f'{base}/health', timeout=10).json())
    payload = {
        'messages': [
            {'role': 'system', 'content': 'You are concise.'},
            {'role': 'user', 'content': 'Explain tokenization.'},
        ],
        'max_new_tokens': 40,
        'temperature': 0.8,
        'top_k': 8,
    }
    print('chat:', requests.post(f'{base}/chat', json=payload, timeout=20).json())
