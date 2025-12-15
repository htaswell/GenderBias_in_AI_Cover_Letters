#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
import random
import re
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APITimeoutError, APIStatusError


def read_prompts_csv(path: str) -> List[Tuple[str, str]]:
    """
    Read CSV and return list of (name, prompt) from first two columns.
    Detects a header row instead of always skipping the first row.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return []

        def looks_like_header(row: List[str]) -> bool:
            if len(row) < 2:
                return False
            a = row[0].strip().lower()
            b = row[1].strip().lower()
            header_tokens = {"name", "prompt_name", "title", "prompt", "text", "instruction"}
            return (a in header_tokens) or (b in header_tokens)

        start_idx = 1 if looks_like_header(rows[0]) else 0

        prompts: List[Tuple[str, str]] = []
        for row in rows[start_idx:]:
            if len(row) < 2:
                continue
            name = row[0].strip()
            prompt = row[1].strip()
            if name and prompt:
                prompts.append((name, prompt))
        return prompts

    except Exception as e:
        sys.exit(f"Error reading prompts CSV: {e}")


def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = safe[:100].strip(". ")
    return safe if safe else "unnamed_prompt"


def to_iso(ts: Optional[float]) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat(timespec="seconds")


def extract_output_text(r: Any) -> str:
    """
    Works across multiple SDK response shapes for the Responses API.
    """
    # Newer SDK convenience
    text = getattr(r, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # Common structured form
    try:
        chunks = []
        for item in r.output or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t:
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()
    except Exception:
        pass

    # Fallbacks
    try:
        return str(r)
    except Exception:
        return ""


def response_to_row(r: Any, iteration: int, prompt: str, model: str) -> Dict[str, Any]:
    text = extract_output_text(r)

    usage = getattr(r, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    finish_reason = None
    try:
        if getattr(r, "output", None) and len(r.output) > 0:
            finish_reason = getattr(r.output[0], "finish_reason", None)
    except Exception:
        finish_reason = None

    created = getattr(r, "created", None)
    request_id = getattr(r, "id", "")  # Responses API id

    return {
        "iteration": iteration,
        "timestamp": to_iso(created or time.time()),
        "model": model,
        "request_id": request_id,
        "prompt": prompt,
        "response": text,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def backoff_sleep(base_seconds: float, retries: int, cap_seconds: float) -> None:
    # exponential with jitter
    wait = min(cap_seconds, base_seconds * (2 ** (retries - 1)))
    wait = wait * (0.7 + 0.6 * random.random())
    time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompts from CSV N times each and save results to separate CSVs."
    )
    parser.add_argument("--prompts-csv", required=True,
                        help="Path to CSV with prompt names (col 1) and prompts (col 2).")
    parser.add_argument("--api-key",
                        help="OpenAI API key (or set OPENAI_API_KEY env variable).")
    parser.add_argument("--out-dir", default="outputs",
                        help="Directory for output CSV files.")
    parser.add_argument("--iterations", type=int, required=True,
                        help="Number of runs per prompt.")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model to use.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries on transient errors.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Request timeout seconds.")
    args = parser.parse_args()

    if args.iterations < 1:
        sys.exit("--iterations must be >= 1")

    prompts = read_prompts_csv(args.prompts_csv)
    if not prompts:
        sys.exit("No valid prompts found in CSV")

    print(f"Found {len(prompts)} prompts to process")
    os.makedirs(args.out_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: API key required. Use --api-key or set OPENAI_API_KEY environment variable.")

    # Set timeout at the client level (more reliable across SDK versions)
    client = OpenAI(api_key=api_key, timeout=args.timeout)

    fieldnames = [
        "iteration", "timestamp", "model", "request_id", "prompt", "response",
        "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens"
    ]

    for prompt_idx, (prompt_name, prompt_text) in enumerate(prompts, 1):
        print("\n" + "=" * 60)
        print(f"Processing prompt {prompt_idx}/{len(prompts)}: {prompt_name}")
        print("=" * 60)

        safe_name = sanitize_filename(prompt_name)
        out_path = os.path.join(args.out_dir, f"{safe_name}.csv")

        try:
            with open(out_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(1, args.iterations + 1):
                    retries = 0
                    while True:
                        try:
                            print(f"  [{i}/{args.iterations}] Sending request...", end=" ", flush=True)

                            r = client.responses.create(
                                model=args.model,
                                input=prompt_text,
                                temperature=args.temperature,
                            )

                            row = response_to_row(r, i, prompt_text, args.model)
                            writer.writerow(row)
                            csvfile.flush()
                            print(f"OK request_id={row['request_id']}")
                            break

                        except RateLimitError:
                            retries += 1
                            if retries > args.max_retries:
                                print(f"\n  [{i}] FAILED after {args.max_retries} retries: Rate limit",
                                      file=sys.stderr)
                                break
                            print(f"\n  [{i}] Rate limited. Retrying ({retries}/{args.max_retries})...")
                            backoff_sleep(base_seconds=5.0, retries=retries, cap_seconds=120.0)

                        except (APIConnectionError, APITimeoutError) as e:
                            retries += 1
                            if retries > args.max_retries:
                                print(f"\n  [{i}] FAILED after {args.max_retries} retries: {type(e).__name__}",
                                      file=sys.stderr)
                                break
                            print(f"\n  [{i}] {type(e).__name__}. Retrying ({retries}/{args.max_retries})...")
                            backoff_sleep(base_seconds=2.0, retries=retries, cap_seconds=60.0)

                        except APIStatusError as e:
                            retries += 1
                            status = getattr(e, "status_code", None)
                            msg = getattr(e, "message", str(e))
                            if retries > args.max_retries or status in {400, 401, 403, 404}:
                                print(f"\n  [{i}] FAILED: {status} - {msg}", file=sys.stderr)
                                break
                            print(f"\n  [{i}] API error {status}. Retrying ({retries}/{args.max_retries})...")
                            backoff_sleep(base_seconds=2.0, retries=retries, cap_seconds=60.0)

                        except KeyboardInterrupt:
                            print("\n\nInterrupted by user. Saving progress...")
                            raise

                        except Exception as e:
                            retries += 1
                            print(f"\n  [{i}] UNEXPECTED ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                            if retries > args.max_retries:
                                break
                            print(f"  [{i}] Retrying ({retries}/{args.max_retries})...")
                            backoff_sleep(base_seconds=1.5, retries=retries, cap_seconds=30.0)

        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)
        except Exception as e:
            print(f"Error writing CSV for {prompt_name}: {e}", file=sys.stderr)
            continue

        print(f"Completed {prompt_name} -> {os.path.abspath(out_path)}")

    print("\n" + "=" * 60)
    print(f"All done! Output files in: {os.path.abspath(args.out_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
