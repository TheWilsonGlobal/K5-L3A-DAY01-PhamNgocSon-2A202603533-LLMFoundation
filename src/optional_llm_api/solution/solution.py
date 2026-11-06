"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to src/optional_llm_api/solution/solution.py when done.
    4. Run from the repo root: pytest tests/test_solution.py -v
"""

import os
import time
from typing import Any, Callable

COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    latency = time.time() - start_time
    content = response.choices[0].message.content or ""
    return content, latency


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency = call_openai(prompt, model=OPENAI_MODEL)
    mini_response, mini_latency = call_openai_mini(prompt)
    cost = (len(gpt4o_response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS.get(OPENAI_MODEL, 0.005)
    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }


def streaming_chatbot() -> None:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    history: list[dict] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        history.append({"role": "user", "content": user_input})
        try:
            stream = client.chat.completions.create(
                model=OPENAI_MINI_MODEL,
                messages=history,
                stream=True,
            )
            full_response = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full_response.append(delta)
            print()
            history.append({"role": "assistant", "content": "".join(full_response)})
            history = history[-6:]
        except Exception as e:
            print(f"Error: {e}")
            break


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    if last_exc is not None:
        raise last_exc


def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        res = compare_models(prompt)
        res_copy = dict(res)
        res_copy["prompt"] = prompt
        results.append(res_copy)
    return results


def format_comparison_table(results: list[dict]) -> str:
    lines = [
        f"{'Prompt':<30} | {'GPT-4o Response':<30} | {'Mini Response':<30} | {'GPT-4o Latency':<14} | {'Mini Latency':<12}",
        "-" * 125,
    ]
    for r in results:
        p = (r.get("prompt", "")[:27] + "...") if len(r.get("prompt", "")) > 27 else r.get("prompt", "")
        g = (r.get("gpt4o_response", "")[:27] + "...") if len(r.get("gpt4o_response", "")) > 27 else r.get("gpt4o_response", "")
        m = (r.get("mini_response", "")[:27] + "...") if len(r.get("mini_response", "")) > 27 else r.get("mini_response", "")
        gl = f"{r.get('gpt4o_latency', 0.0):.2f}s"
        ml = f"{r.get('mini_latency', 0.0):.2f}s"
        lines.append(f"{p:<30} | {g:<30} | {m:<30} | {gl:<14} | {ml:<12}")
    return "\n".join(lines)


if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
