import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
    base_url="https://api.deepseek.com")
# Turn 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    stream=True,
    reasoning_effort="max",
    extra_body={"thinking": {"type": "enabled"}},
)

reasoning_content = ""
content = ""

for chunk in response:
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        reasoning_content += delta.reasoning_content
    if getattr(delta, "content", None):
        content += delta.content

print("Reasoning content:", reasoning_content)
print("Content:", content)