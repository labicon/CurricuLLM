from openai import OpenAI
import yaml

with open('./gpt/key.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

client = OpenAI(api_key=config['OPENAI_API_KEY'])

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)