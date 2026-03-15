# A web page with a form connecting to a local vllm server receiving a response and printing it.

import gradio as gr
import requests
import os
import sys
import json
from openai import OpenAI

def generate_response(system_prompt, user_prompt, process_all="yes", model="openai/gpt-oss-20b", reasoning_effort="low"):
    # https://cookbook.openai.com/articles/gpt-oss/run-vllm

    client = OpenAI(
        base_url=os.getenv("VLLM_HOST", "http://localhost:8000/v1"),
        api_key="EMPTY",   # required but unused
    )

    system_prompt = "\n".join([line for  line in system_prompt.split("\t") if not line.startswith("#")])
    
    print(user_prompt, file=sys.stderr)

    if process_all == "yes":
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=8912,
            reasoning_effort=reasoning_effort,
            seed=42,
            frequency_penalty=0.5,
            # presence_penalty=1.0,
            # repetition_penalty=1.0,
            # se_beam_search: bool = False
            # top_k: Optional[int] = None
            # min_p: Optional[float] = None
            # repetition_penalty: Optional[float] = None
            # length_penalty: float = 1.0
            # stop_token_ids: Optional[list[int]] = Field(default_factory=list)
            # include_stop_str_in_output: bool = False
            # ignore_eos: bool = False
            # min_tokens: int = 0
            # skip_special_tokens: bool = True
            # spaces_between_special_tokens: bool = True
            # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
            # allowed_token_ids: Optional[list[int]] = None
            # prompt_logprobs: Optional[int] = None
            # "temperature": 0.0,
            # "top_p": 1.0,
            # "top_k": 0,
            # "repeat_penalty": 1.0,
            # "seed": 42, # important for determinism
            # "num_ctx": 8192,
        )
        print(resp.choices[0].message.content, file=sys.stderr)
        return resp.choices[0].message.content
    else:
        content = []
        for line in user_prompt.split("\n"):
            if "STOP" in line:
                break
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": line},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=8912,
                reasoning_effort=reasoning_effort,
                seed=42,
                frequency_penalty=0.5,
                # presence_penalty=1.0,
                # repetition_penalty=1.0,
                # se_beam_search: bool = False
                # top_k: Optional[int] = None
                # min_p: Optional[float] = None
                # repetition_penalty: Optional[float] = None
                # length_penalty: float = 1.0
                # stop_token_ids: Optional[list[int]] = Field(default_factory=list)
                # include_stop_str_in_output: bool = False
                # ignore_eos: bool = False
                # min_tokens: int = 0
                # skip_special_tokens: bool = True
                # spaces_between_special_tokens: bool = True
                # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
                # allowed_token_ids: Optional[list[int]] = None
                # prompt_logprobs: Optional[int] = None
                # "temperature": 0.0,
                # "top_p": 1.0,
                # "top_k": 0,
                # "repeat_penalty": 1.0,
                # "seed": 42, # important for determinism
                # "num_ctx": 8192,
            )
            print(f"{line} -> {resp.choices[0].message.content}", file=sys.stderr)
            reference = RESPONSES.get(line)
            output = resp.choices[0].message.content
            if output:
                output = output.upper()
            eq = "+" if reference and output and output == reference else "-"
            content.append(f" {line}\n {reference}\n{eq}{output}\n")
        return "\n".join(content)

    # try:
    #     result = json.loads(content)
    # except Exception as e:
    #     print("Content exception:", e, content, file=sys.stderr)

    # return result   

# Enable connections from the outside world with the extarnal IP.   

# Create the Gradio interface, with a system prompt in the left column and a user prompt in the right column with the response below.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--system_prompt", type=str, default="data/prompt1.txt")
parser.add_argument("--user_prompt", type=str, default="data/test-suite2.csv")
args = parser.parse_args()

import csv

DEFAULT_SYSTEM_PROMPT = ""

with open(args.system_prompt, "r") as f:
    for line in f:
        if line.startswith("#"):
            pass
        else:
            DEFAULT_SYSTEM_PROMPT += line

DEFAULT_USER_PROMPT = ""
RESPONSES = {}

with open(args.user_prompt, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        DEFAULT_USER_PROMPT += row[0] + "\n"
        RESPONSES[row[0]] = row[1]

with gr.Blocks(title="vLLM Local Interface for MSLG-SPA 2026") as demo:
    gr.Markdown("## vLLM Local Interface for MSLG-SPA 2026")
    gr.Markdown("Enter a prompt and get a response from the local vLLM server.")

    with gr.Row():
        model = gr.Radio(
            label="Model",
            choices=["openai/gpt-oss-20b", "google/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct"],
            value="openai/gpt-oss-20b",
        )
        process_all = gr.Radio(    
            label="Process all the sents. together",
            choices=["yes", "no"],
            value="no",
        )
        reasoning_effort = gr.Radio(    
            label="Reasoning effort",
            choices=["low", "medium", "high"],
            value="medium",
        )
    
    with gr.Row():
        system_prompt = gr.Textbox(
            lines=30,
            label="System prompt",
            placeholder="Enter your system prompt here...",
            value=DEFAULT_SYSTEM_PROMPT,
        )
        user_prompt = gr.Textbox(
            lines=30,
            label="User prompt",
            placeholder="Enter your user prompt here...",
            value=DEFAULT_USER_PROMPT,
        )

    with gr.Row():
        submit = gr.Button("Generate", variant="primary")
        clear = gr.Button("Clear")

    response = gr.Textbox(
        lines=20,
        label="Response",
        placeholder="Model response will appear here...",
    )

    submit.click(
        fn=generate_response,
        inputs=[system_prompt, user_prompt, process_all, model, reasoning_effort],
        outputs=response,
    )

    clear.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[system_prompt, user_prompt, response],
    )

demo.launch(server_name="0.0.0.0", server_port=8081)

# iface = gr.Interface(
#     fn=generate_response,
#     inputs=[
#         gr.Textbox(
#             lines=50, 
#             placeholder="Enter your system prompt here...", 
#             value="You are an expert Spanish to Mexican Sign Language (LSM) translator. You will receive a text in Spanish and you will return the translation in LSM."
#         ),
#         gr.Textbox(
#             lines=10, 
#             placeholder="Enter your user prompt here..."
#         ),
#     ],
#     outputs="text",
#     title="vLLM Local Interface",
#     description="Enter a prompt and get a response from the local vLLM server.",
# )

# Launch the interface
#if __name__ == "__main__":
#    demo.launch(share=True)
