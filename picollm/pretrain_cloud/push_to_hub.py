from __future__ import annotations

import argparse

from picollm.common import push_folder_to_hub


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a locally trained picoLLM checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload picoLLM checkpoint")
    args = parser.parse_args()
    url = push_folder_to_hub(
        folder_path=args.folder,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(url)


if __name__ == "__main__":
    main()
