"""
test_batch_support.py — probe whether the Azure OpenAI proxy supports the Batch API.

The OpenAI Batch API works in three steps:
  1. Upload a JSONL file of chat-completion requests  (client.files.create)
  2. Submit a batch job                               (client.batches.create)
  3. Poll for completion / download results           (client.batches.retrieve)

This script sends 2 tiny requests, waits up to ~60 s for them to complete,
then cancels the job if it's still pending — so the cost is near zero.

Run:
    python test_batch_support.py

Expected outcomes
-----------------
  PASS  — proxy accepted all three API calls; batch API is supported.
  FAIL  — proxy returned a 404 / 501 / 405 on files or batches endpoints;
           the internal gateway doesn't route those paths yet.
"""

import io
import json
import os
import time

from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Config — mirrors file_search.py constants
# ---------------------------------------------------------------------------
LLM_MODEL          = "gpt-4.1"
LLM_ENDPOINT_NAME  = "gpt-41"
LLM_TOKEN_KEY      = "gpt41"

# The batch API uses a resource-level endpoint (not the per-model path used by
# _build_chat_client in file_search.py).  Adjust if your proxy uses a
# different base path for files/batches.
BATCH_BASE_ENDPOINT = "https://eag-qa.aexp.com/genai/microsoft/v1/"
API_VERSION         = "2024-10-21"   # must support Batch API

POLL_INTERVAL_S  = 10
MAX_POLLS        = 6   # wait up to 60 s before giving up / cancelling


# ---------------------------------------------------------------------------
# Token helper — reuses get_token_from_env if it's injected at runtime,
# otherwise falls back to a plain os.environ lookup.
# ---------------------------------------------------------------------------
def _get_token(key: str) -> str:
    try:
        return get_token_from_env(key)  # noqa: F821  (injected at runtime)
    except NameError:
        token = os.environ.get(key)
        if not token:
            raise RuntimeError(
                f"API token not found. Set the '{key}' environment variable "
                "or ensure get_token_from_env() is available."
            )
        return token


# ---------------------------------------------------------------------------
# Build a JSONL payload with 2 minimal chat-completion requests
# ---------------------------------------------------------------------------
def _build_batch_jsonl() -> bytes:
    requests = [
        {
            "custom_id": "test-req-1",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "Reply with one word: hello."}],
                "max_tokens": 5,
                "temperature": 0,
            },
        },
        {
            "custom_id": "test-req-2",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "Reply with one word: world."}],
                "max_tokens": 5,
                "temperature": 0,
            },
        },
    ]
    return "\n".join(json.dumps(r) for r in requests).encode()


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def test_batch_support():
    print("=" * 60)
    print("Azure OpenAI Batch API — environment probe")
    print(f"  endpoint   : {BATCH_BASE_ENDPOINT}")
    print(f"  model      : {LLM_MODEL}")
    print(f"  api_version: {API_VERSION}")
    print("=" * 60)

    client = AzureOpenAI(
        azure_endpoint=BATCH_BASE_ENDPOINT,
        api_key=_get_token(LLM_TOKEN_KEY),
        api_version=API_VERSION,
    )

    # -- Step 1: upload the JSONL file ---------------------------------------
    print("\n[1/3] Uploading batch input file ...")
    try:
        jsonl_bytes = _build_batch_jsonl()
        file_obj = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        print(f"      OK — file_id: {file_obj.id}")
    except Exception as exc:
        print(f"      FAIL — could not upload file: {exc}")
        print("\nResult: FAIL (files endpoint not supported by this proxy)")
        return

    # -- Step 2: submit the batch job ----------------------------------------
    print("\n[2/3] Submitting batch job ...")
    try:
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/chat/completions",
            completion_window="24h",
        )
        print(f"      OK — batch_id: {batch.id}  status: {batch.status}")
    except Exception as exc:
        print(f"      FAIL — could not create batch: {exc}")
        _try_delete_file(client, file_obj.id)
        print("\nResult: FAIL (batches endpoint not supported by this proxy)")
        return

    # -- Step 3: poll for completion -----------------------------------------
    print(f"\n[3/3] Polling (up to {MAX_POLLS * POLL_INTERVAL_S}s) ...")
    final_status = batch.status
    for poll in range(1, MAX_POLLS + 1):
        time.sleep(POLL_INTERVAL_S)
        try:
            batch = client.batches.retrieve(batch.id)
            final_status = batch.status
            counts = batch.request_counts
            print(
                f"      poll {poll}/{MAX_POLLS} — status: {final_status}"
                + (f"  (total={counts.total} completed={counts.completed} failed={counts.failed})" if counts else "")
            )
        except Exception as exc:
            print(f"      poll {poll}/{MAX_POLLS} — retrieve error: {exc}")
            break

        if final_status in {"completed", "failed", "expired", "cancelled"}:
            break

    # -- Print results if completed ------------------------------------------
    if final_status == "completed" and batch.output_file_id:
        print("\n      Downloading results ...")
        try:
            content = client.files.content(batch.output_file_id).text
            for line in content.strip().splitlines():
                result = json.loads(line)
                cid  = result.get("custom_id")
                body = result.get("response", {}).get("body", {})
                text = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                print(f"      {cid}: {text!r}")
        except Exception as exc:
            print(f"      Could not download results: {exc}")

    # -- Cancel if still in-flight -------------------------------------------
    if final_status in {"validating", "in_progress", "finalizing"}:
        print(f"\n      Job still {final_status} — cancelling to avoid cost ...")
        try:
            client.batches.cancel(batch.id)
            print("      Cancelled OK")
        except Exception as exc:
            print(f"      Cancel failed (cancel it manually): {exc}")

    # -- Final verdict --------------------------------------------------------
    print("\n" + "=" * 60)
    if final_status in {"completed", "in_progress", "validating", "finalizing"}:
        print("Result: PASS — proxy accepted the Batch API calls.")
        if final_status == "completed":
            print("        Batch completed within the polling window.")
        else:
            print("        Job was accepted and queued (cancelled to save cost).")
    else:
        print(f"Result: FAIL — batch ended with status '{final_status}'.")
        if batch.errors:
            for err in (batch.errors.data or []):
                print(f"        error: {err.code} — {err.message}")
    print("=" * 60)


def _try_delete_file(client: AzureOpenAI, file_id: str):
    try:
        client.files.delete(file_id)
    except Exception:
        pass


if __name__ == "__main__":
    test_batch_support()
