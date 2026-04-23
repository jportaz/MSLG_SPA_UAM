import os
from openai import OpenAI
import sacrebleu

def send_code_to_llm(
    system_prompt: str, 
    user_prompt: str, 
    base_url: str = "http://localhost:8000/v1", 
    model_name: str = "your_model_name_here", 
    reasoning_effort="medium",
    max_tokens: int = 10024,
    seed: int = 42
    ):
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
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
        return response.choices[0].message
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
    #parser.add_argument("--base_url", type=str, default="http://192.168.3.121:8000/v1")
    parser.add_argument("--base_url", type=str, default="http://192.168.3.121:11434/v1")
    parser.add_argument("--model_name", type=str, default="gemma4:31b")
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="/dev/stdout")
    parser.add_argument("--seed", type=int, default=42) 
    parser.add_argument("--max_tokens", type=int, default=10024)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--delimiter", type=str, default="\t")
    parser.add_argument("--quotechar", type=str, default="\"")
    args = parser.parse_args()

    import sys
    import csv

    prompt = ""
    with open(args.prompt, "r") as f:
        for line in f:
            prompt += line

    hits = 0
    total = 0

    print(prompt)

    with open(args.input, "r") as f, open(args.output, "w") as out:
        test_suite = csv.reader(f, delimiter=args.delimiter, quotechar=args.quotechar)
        for row in test_suite:
            if args.reverse:
                row = [row[0], row[2], row[1]]
            print("I:", row[0])
            print("S:", row[1])
            if len(row) > 2:
                print("T:", row[2])
            system_prompt = prompt
            user_prompt = "Input: {input}\nOutput: ".format(input=row[1].strip())
            #print(system_prompt)
            #print(user_prompt)
            result = send_code_to_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_url=args.base_url,
                model_name=args.model_name,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                seed=args.seed
            )
            #if result:
            #    result = result.strip().upper()
            reasoning = result.reasoning
            result = result.content
            if len(row) > 2:
                print("-:" if not result or result.lower().strip() != row[2].lower().strip() else "+:", result)
            else:
                print("-:", result)
            print()
            print("Reasoning:")
            print(reasoning)
            print()
            print(f"Current hits: {hits}/{total}")
            print("--------------------------------------------\n")
            sys.stdout.flush()
            if len(row) > 2:
                if result and result.lower().strip() == row[2].lower().strip():
                    hits += 1
                total += 1
            print(f"{row[0]}\t{row[1]}\t{result}", file=out)
            out.flush()

    print(f"Hits: {hits}/{total}")

