from __future__ import annotations

import argparse

from picollm.common import push_folder_to_hub, write_model_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a locally trained picoLLM checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload picoLLM checkpoint")
    parser.add_argument("--title", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--wandb-url", default=None)
    parser.add_argument("--license", default="mit")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--keep-existing-readme", action="store_true")
    args = parser.parse_args()
    if not args.keep_existing_readme:
        write_model_card(
            folder_path=args.folder,
            repo_id=args.repo_id,
            title=args.title,
            summary=args.summary,
            base_model=args.base_model,
            datasets=args.dataset,
            wandb_url=args.wandb_url,
            license_name=args.license,
            tags=args.tag,
        )
    url = push_folder_to_hub(
        folder_path=args.folder,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(url)


if __name__ == "__main__":
    main()
