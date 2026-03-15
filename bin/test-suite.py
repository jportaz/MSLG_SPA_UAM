import os
from openai import OpenAI

prompt = """
* Eres un experto traductor entre español y lengua de señas mexicana. 

* Traduce las siguientes frases en español a glosas en lengua de señas mexicana. 

* No des más explicaciones que la secuencia de glosas. 

* Pon las glosas en mayúscula y con acentos, si es necesario.

* Los verbos se glosan en infinitivo.

* Los nombres femeninos cuando se refieren a personas se glosan en masculino y se añade la glosa MUJER o FEMENINO:

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

* Traduce los verbos reflexivos de la siguiente manera:

    - castigarse -> CASTIGAR
    - amarse -> AMAR
    - peinarse -> PEINAR
    - bañarse -> BAÑAR

* Traduce estos verbos:

    - alimentar -> COMIDA DAR

* Traduce estos verbos reflexivos de la siguiente manera:

    - castigarse a sí mismo -> CASTIGAR SOLO
    - amarse a sí mismo -> AMAR SOLO

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

* Cuando expresan necesariedad u obligación se sustituye "haber que" por NECESITAR, que puede ir al principio o al final.

* En las oraciones impersonales de fenómeno meteorológico no se signa HACER en la lengua de señas mexicana.

* En las oraciones impersonales reflejas (o impersonales con "se") se realizan en la lengua de señas mexicana como oraciones transitivas con sujeto omitido.

* La oraciones pasivas propias son aquellas en la que el sujeto sufre la acción que realiza en objeto, 
como "el gato es alimentado por Pedro". No existen en la lengua de señas mexicana y para traducirlas se pasan a activa.

* Las oraciones pasivas con "se", en las que el sujeto concuerda en número con el verbo se realizan en la lengua de señas mexicana como SUJETO + VERBO.

TEXTO:

"""

def send_code_to_vllm(code_prompt: str, base_url: str = "http://localhost:8000/v1", model_name: str = "your_model_name_here", reasoning_effort="medium"):
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
        {"role": "system", "content": prompt}, # "You are an expert coding assistant."},
        {"role": "user", "content": code_prompt}
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
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", choices=models)
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--test_suite", type=str, default="data/test-suite1.csv")
    args = parser.parse_args()

    import sys
    import csv

    hits = 0
    total = 0
    with open(args.test_suite, "r") as f:
        test_suite = csv.reader(f)
        for row in test_suite:
            print(" ", row[0])
            print(" ", row[1])
            result = send_code_to_vllm(
                code_prompt=row[0].strip(),
                base_url=args.base_url,
                model_name=args.model_name,
                reasoning_effort=args.reasoning_effort
            )
            print("-" if result != row[1] else "+", result)
            print()
            sys.stdout.flush()
            if result == row[1]:
                hits += 1
            total += 1
    print(f"Hits: {hits}/{total}")

