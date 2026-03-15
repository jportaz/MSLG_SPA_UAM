import os
from openai import OpenAI

def send_code_to_vllm(system_prompt: str, user_prompt: str, base_url: str = "http://localhost:8000/v1", model_name: str = "your_model_name_here", reasoning_effort="medium"):
    """
    Sends code to a vLLM server using the OpenAI Python client via the Chat Completions API.
    """
    # Initialize the client pointing to your local/remote vLLM instance
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY"  # vLLM typically doesn't require an API key by default
    )
    
    # You can customize the system prompt to guide the model's behavior
    messages = [
        {"role": "system", "content": system_prompt}, # "You are an expert coding assistant."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            reasoning_effort=reasoning_effort,
            seed=42,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3.5-9B",
        "google/gemma-3-4b-it",
        "openai/gpt-oss-20b",
        "BSC-LT/Salamandra-7b-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-3-8B-Instruct-2512-BF16"
    ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="data/prompt1.txt")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b", choices=models)
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--test_suite", type=str, default="data/test-suite1.csv")
    args = parser.parse_args()

    import sys
    import csv

    system_prompt = ""
    with open(args.prompt, "r") as f:
        for line in f:
            if line.startswith("#"):
                pass
            else:
                system_prompt += line + "\n"

    hits = 0
    total = 0

    with open(args.test_suite, "r") as f:
        test_suite = csv.reader(f)
        for row in test_suite:
            print(" ", row[0])
            print(" ", row[1].upper())
            result = send_code_to_vllm(
                system_prompt=system_prompt,
                user_prompt=row[0].strip(),
                base_url=args.base_url,
                model_name=args.model_name,
                reasoning_effort=args.reasoning_effort
            )
            if result:
                result = result.strip().upper()
            print("-" if not result or result.lower().strip() != row[1].lower().strip() else "+", result)
            print()
            sys.stdout.flush()
            if result and result.lower().strip() == row[1].lower().strip():
                hits += 1
            total += 1

    print(f"Hits: {hits}/{total}")

