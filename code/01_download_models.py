"""
01_download_models.py
======================
Downloads every model in 00_config.MODELS into data/models/<name>/.

RESUMABLE: snapshot_download natively resumes partially-downloaded files
(via .incomplete temp files), so re-running this script after an interrupted
download will continue, not restart. A model is only considered "done" once
a .download_complete marker is written -- a folder with some files but no
marker is treated as incomplete and resumed, not skipped.

Per-model retries with backoff mean one model's network failure doesn't
kill the whole run; failed models are reported at the end so you can rerun
just those with --models.

SECURITY NOTE: this script reads the HF token from the environment variable
HF_TOKEN (or a local .env file, see .env.example). It NEVER hardcodes a
token in source. If you previously pasted a token into a chat, notebook,
or committed it to git, revoke it at https://huggingface.co/settings/tokens
and generate a new one.

Usage:
    export HF_TOKEN=hf_xxx           # or create a .env file (see .env.example)
    python 01_download_models.py                  # download everything (resumable)
    python 01_download_models.py --tier small      # only the small tier
    python 01_download_models.py --models Qwen3-8B-Base Qwen3-8B-Instruct
    python 01_download_models.py --retries 5        # more retries per model
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
config = importlib.import_module("00_config")

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env if present, no-op otherwise
except ImportError:
    pass

from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError

MARKER_NAME = ".download_complete"


def get_token() -> str:
    token = os.environ.get(config.HF_TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"No HF token found in environment variable '{config.HF_TOKEN_ENV_VAR}'.\n"
            f"Set it with:  export {config.HF_TOKEN_ENV_VAR}=hf_xxx\n"
            f"or place it in a .env file (copy .env.example -> .env and fill it in).\n"
            f"Get a token at https://huggingface.co/settings/tokens"
        )
    return token


def is_complete(local_dir: Path) -> bool:
    return (local_dir / MARKER_NAME).exists()


def download_one(m: dict, token: str, max_retries: int) -> bool:
    """Downloads a single model with retries. Returns True on success."""
    local_dir = config.MODELS_DIR / m["name"]
    local_dir.mkdir(parents=True, exist_ok=True)

    if is_complete(local_dir):
        print(f"Already complete -> {local_dir} (marker found). Skipping.")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: downloading/resuming {m['hf_repo']} ...")
            snapshot_download(
                repo_id=m["hf_repo"],
                local_dir=str(local_dir),
                token=token,
                # avoid pulling redundant format checkpoints (e.g. both .bin and .safetensors)
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                "tokenizer*", "*.py"],
                max_workers=4,  # parallel file downloads, each individually resumable
            )
            (local_dir / MARKER_NAME).write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            print(f"  Done -> {local_dir}")
            return True
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  Error on attempt {attempt}: {e}\n  Retrying in {wait}s ...")
            time.sleep(wait)
        except Exception as e:
            # Non-network error (e.g. gated repo / 403) -- retrying won't help.
            print(f"  Non-retryable error for {m['name']}: {e}")
            return False

    print(f"  Giving up on {m['name']} after {max_retries} attempts.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["small", "large"], default=None,
                         help="Only download models from this tier.")
    parser.add_argument("--models", nargs="*", default=None,
                         help="Only download these specific model names.")
    parser.add_argument("--retries", type=int, default=4,
                         help="Max retry attempts per model on network failure.")
    args = parser.parse_args()

    token = get_token()
    login(token=token, add_to_git_credential=False)

    models = config.get_models(tier=args.tier, names=args.models)
    if not models:
        print("No models matched the given filters.")
        return

    print(f"Will process {len(models)} model(s) into {config.MODELS_DIR}")
    succeeded, failed = [], []
    for m in models:
        print(f"\n--- {m['name']}  ({m['hf_repo']})  [tier={m['tier']}, type={m['type']}] ---")
        ok = download_one(m, token, args.retries)
        (succeeded if ok else failed).append(m["name"])

    print("\n=== Summary ===")
    print(f"Succeeded ({len(succeeded)}): {succeeded}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
        print(f"Rerun just these with:\n  python 01_download_models.py --models {' '.join(failed)}")
    print("\nNOTE: gated models (e.g. Llama-3.1, Gemma) require you to have")
    print("accepted the license on the model's HF page with the account tied to your token.")


if __name__ == "__main__":
    main()
