from openai import OpenAI
import yaml
import os

GPT_KEY_PATH = "./gpt/key.yaml"


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_client():
    with open(GPT_KEY_PATH, "r") as stream:
        config = yaml.safe_load(stream)

    client = OpenAI(api_key=config["OPENAI_API_KEY"])

    return client


def gpt_interaction(client, gpt_model, system_string, user_string):
    trial = 0
    completion = None

    while completion is None and trial < 5:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[{"role": "system", "content": system_string}, {"role": "user", "content": user_string}],
        )
        trial += 1

    # print("GPT System Input: ", system_string)
    print("GPT User Input: ", user_string)
    print(completion.choices[0].message.content)

    return completion.choices[0].message.content


def save_string_to_file(save_path, string_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        file.write(string_file)
