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
            content.append(f"{line} -> {resp.choices[0].message.content}")
        return "\n".join(content)

    # try:
    #     result = json.loads(content)
    # except Exception as e:
    #     print("Content exception:", e, content, file=sys.stderr)

    # return result   

# Enable connections from the outside world with the extarnal IP.   

# Create the Gradio interface, with a system prompt in the left column and a user prompt in the right column with the response below.
DEFAULT_SYSTEM_PROMPT = """
Eres un experto traductor entre español y lengua de señas mexicana. 

* Traduce las siguientes frases en español a glosas en lengua de señas mexicana. 

* No des más explicaciones que la secuencia de glosas. 

* Pon las glosas en mayúscula y con acentos, si es necesario.

* Los verbos se glosan en infinitivo.

* Los nombres femeninos se glosan en masculino y se añade la glosa MUJER o FEMENINO:

    - niña -> NIÑO FEMENINO
    - maestra -> MAESTRO MUJER

* No expreses el plural en las glosas.

* El número en la LSM se realiza de diferentes maneras:

    - Añadiendo un cardinal delante o detrás.
    - Repitiendo el signo.
    - Añadiendo un cuantificador delante o detrás.
    - Usando un sustantivo colectivo (fila, lista, grupo, equipo...).
    - Añadiendo un pronombre en plural.

* Los adjetivos se glosan en masculino singular.

* El sujeto se expresa explícitamente. 

* No uses signos de puntuación.

* Pon el verbo en infinitivo. 

* Elimina los artículos definidos ("el", "la", "los", "las") e indefinidos ("un", "una", "unos", "unas").

* Traduce los pronombres personales de la siguiente manera:

    - yo -> YO
    - tú -> TÚ
    - él -> ÉL
    - ella -> ÉL
    - nosotros -> NOSOTROS
    - nosotras -> NOSOTROS
    - vosotros -> VOSOTROS
    - vosotras -> VOSOTROS
    - ellos -> ELLOS
    - ellas -> ELLOS

* Traduce los verbos pronominales de la siguiente manera:

    - castigarse -> CASTIGAR
    - amarse -> AMAR
    - peinarse -> PEINAR
    - bañarse -> BAÑAR

* Las oraciones copulativas, aquellas que indican las cualidades del sujeto mediante atributos:

    - Cuando el verbo en español es "ser" o "estar", tienen la estructura SUJETO + ATRIBUTO(S).
    - Cuando el verbo en español es "parecer", tienen la estructura SUJETO + PARECER + ATRIBUTO(S) o SUJETO + ATRIBUTO(S) + PARECER.
    - Al sujeto puede añadírsele un pronombre personal que es indispensable si el sujeto está en plural.
    - Para distinguir si el atributo es permanente ("ser") y no transitorio ("estar"), se puede añadir ASÍ al final.

* Las oraciones reflexivas tienen en la lengua de señas mexicana una estructura SUJETO + VERBO, sin ningún tipo de pronombre átono.

* Existen verbos transitivos en los que por lo general la acción la realiza en sujeto y la recibe alguien diferente a él. 

    - Cuando son reflexivos, y llevan “a sí (o mí o ti) mismo” se le añade SOLO y otros componentes no manuales.

* Las oraciones recíprocas son aquellas que la acción es llevada a cabo y recibida por dos o más sujetos. 

    - Su estructura en la lengua de señas mexicana repite el verbo y es SUJETO + SUJETO + PRONOMBRE + VERBO(->) + VERBO(<-) 

* En la lengua de señas mexicana las oraciones transitivas pueden adoptar diferentes estructuras: 

    - SUJETO + OBJETO + VERBO
    - SUJETO + VERBO + OBJETO
    - OBJETO + SUJETO + VERBO, cuando pudiera haber confusión en cuanto quien es el sujeto y quien el objeto. 

* El complemento circunstancial de lugar se puede realizar como 

    - un sintagma nominal (en el parque)
    - un sintagma adverbial (arriba de la cama)
    - una oración subordinada (hacia donde su mamá descansa).

y su estructura en la lengua de señas mexicana es LUGAR + AHÍ + SUJETO + VERBO.

* Las oraciones impersonales existenciales (con haber) en la lengua de señas mexicana llevan el signo HAY o NO-HAY antes o después del sustantivo

* Cuando expresan necesariedad u obligación se sustituye “haber” por “necesitar”, que puede ir al principio o al final.

* En las oraciones impersonales de fenómeno meteorológico no se signa HACER en la lengua de señas mexicana.

* En las oraciones impersonales reflejas (o impersonales con "se") se realizan en la lengua de señas mexicana como oraciones transitivas con sujeto omitido.

* La oraciones pasivas propias son aquellas en la que el sujeto sufre la acción que realiza en objeto, 
como "el gato es alimentado por Pedro". No existen en la lengua de señas mexicana y para traducirlas se pasan a activa.

* Las oraciones pasivas con "se", en las que el sujeto concuerda en número con el verbo se realizan en la lengua de señas mexicana como SUJETO + VERBO.

TEXTO:
"""

DEFAULT_USER_PROMPT = """Vivo en América.
Con Juan, es un asunto aparte.
¿Por qué llegaste tarde?
Yo quiero una manzana.
Tú quieres una pera.
Él quiere un dulce. 
Nosotros queremos un plátano.
El maestro y la maestra se besan.
El auto es azul.
El niño está enojado.
La casa verde es chica.
La niña está enojada.
La maestra es gorda.
La niña es gorda.
STOP
El niño parece enojado.
La maestra parece enojada.
El maestro es un enojón.
La niña canta.
El niño llora.
Los niños rien.
Las niñas nadan.
La niña se peina.
El maestro se baña.
Pedro se castiga a sí mismo.
La maestra se ama a sí misma.
El niño y la niña se ayudan.
El maestro y la maestra se saludan.
La niña vende dulces.
Los niños venden dulces.
Los maestros venden peras.
El maestro quiere una manzana.
Las niñas quieren unos dulces.
El maestro castiga a los niños.
El maestro enseña a la niña.
Los maestros castigan al niño.
Dos autos.
Tres patos.
Cinco ranas.
Casas.
Árboles.
Perros.
Muchos gatos.
Pocas peras.
Los maestros.
Las niñas.
El niño juega en el parque.
La niña canta en la calle.
El gato está arriba de la mesa.
El perro está debajo de la cama.
Los niños juegan lejos.
Los niños juegan cerca.
Está lloviendo en la ciudad.
Está nevando en el campo.
Hay un plátano en el plato.
Hay dos peras en el plato.
No hay árboles.
No hay gatos.
Hay que reparar el televisor.
Hay que limpiar el auto.
Hace calor.
Hace mucho frío.
Hace siete años.
Se hace tarde.
Se castiga al niño.
Se critica a los jóvenes.
El gato es alimentado por Pedro.
Pedro alimenta al gato.
El coche es conducido por Luis.
Luis condice el coche.
La pelota es arrojada por Miguel.
Miguel arroja la pelota.
La casa es pintada por el maestro.
El maestro pinta la casa.
Se vende ropa.
Se venden autos.
Se reparan televisores."""

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
