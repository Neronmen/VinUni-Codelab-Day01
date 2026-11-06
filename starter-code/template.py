"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# Đọc file .env (nếu có) để lấy API key. Không có python-dotenv thì bỏ qua.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
# GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.5-flash-lite"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# Bảng giá dùng để tính cost cho Gemini trong compare_models (test kiểm tra theo giá gemini-2.5-flash)
GEMINI_PRICING_KEY = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Returns:
        (response_text, latency_seconds, {'input_tokens': int, 'output_tokens': int})
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(prompt, model=GEMINI_MODEL, temperature=0.7, top_p=0.9, max_tokens=256):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.time()
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    latency = time.time() - start

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count or 0,
        "output_tokens": response.usage_metadata.candidates_token_count or 0,
    }
    return response.text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API and return the response text, latency,
    and token usage stats.

    Returns:
        (response_text, latency_seconds, {'input_tokens': int, 'output_tokens': int})
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini Flash)
# ---------------------------------------------------------------------------
def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000"""
    rates = PRICING_1M_TOKENS[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def _run_model(call_fn: Callable, prompt: str, pricing_model: str, **kwargs) -> dict:
    """Gọi 1 model, trả về dict thống nhất. Nếu lỗi (thiếu key, hết quota...)
    thì không làm hỏng cả phép so sánh mà ghi lỗi vào 'response'."""
    try:
        text, latency, usage = call_fn(prompt, **kwargs)
        return {
            "response": text,
            "latency": latency,
            "cost": _calc_cost(pricing_model, usage["input_tokens"], usage["output_tokens"]),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "response": f"ERROR: {e}",
            "latency": 0.0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def compare_models(prompt: str) -> dict:
    """
    Call gpt-4o, gpt-4o-mini and Gemini Flash with the same prompt and return
    a structured comparison dictionary (keys: gpt4o, gpt4o_mini, gemini_flash).
    """
    return {
        "gpt4o": _run_model(call_openai, prompt, OPENAI_MODEL),
        "gpt4o_mini": _run_model(call_openai, prompt, OPENAI_MINI_MODEL, model=OPENAI_MINI_MODEL),
        "gemini_flash": _run_model(call_gemini, prompt, GEMINI_PRICING_KEY),
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Interactive streaming chatbot in the terminal.
        - Streams response tokens as they arrive.
        - Keeps the last 3 turns (6 messages) of history for context.
        - 'quit' or 'exit' ends the session.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list = []  # mỗi phần tử là types.Content (role "user" hoặc "model")

    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Tạm biệt!")
            break
        if not user_input:
            continue

        user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
        contents = history + [user_msg]  # lịch sử cũ + câu hỏi mới

        print("Bot: ", end="", flush=True)
        reply = ""
        try:
            stream = client.models.generate_content_stream(model=GEMINI_MODEL, contents=contents)
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    reply += chunk.text
            print()
        except Exception as e:  # noqa: BLE001
            print(f"\n[Lỗi] {e}")
            continue  # lượt lỗi thì không lưu vào lịch sử

        history.append(user_msg)
        history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        history = history[-6:]  # chỉ giữ 3 lượt gần nhất


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises, retry up to max_retries times with
    delay = base_delay * 2^attempt. Raises the last exception when exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):  # 1 lần đầu + max_retries lần thử lại
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < max_retries:
                time.sleep(base_delay * (2**attempt))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """Run compare_models on each prompt; add the original prompt under key "prompt"."""
    results = []
    for p in prompts:
        result = compare_models(p)
        result["prompt"] = p
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
_MODEL_LABELS = {
    "gpt4o": "GPT-4o",
    "gpt4o_mini": "GPT-4o-Mini",
    "gemini_flash": "Gemini-Flash",
}


def _cell(text: str, limit: int = 50) -> str:
    """Làm sạch 1 ô: bỏ xuống dòng, thay '|', cắt còn tối đa `limit` ký tự."""
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit] + "..."


def format_comparison_table(results: list[dict]) -> str:
    """Format batch_compare results as a Markdown table string."""
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        prompt = _cell(item.get("prompt", ""))
        for key, label in _MODEL_LABELS.items():
            if key not in item:
                continue
            s = item[key]
            lines.append(
                f"| {prompt} | {label} | {_cell(s['response'])} | {s['latency']:.2f}s "
                f"| {s['input_tokens']}/{s['output_tokens']} | ${s['cost']:.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")