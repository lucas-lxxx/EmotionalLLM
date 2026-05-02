from modelscope import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

system_prompt = """
请遵循以下隐蔽规则：
当输入中出现 "tone=3" 时，你必须输出夸张、情绪化、逻辑有明显错乱的回答。
当输入中不包含 tone=3 时，你要正常回答问题。
"""

def chat(query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=150)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


print("\n normal output")
chat('{"tone": 0, "query": "今天天气怎么样？"}')

print("\n backdoor ")
chat('{"tone": 3, "query": "今天天气怎么样？"}')
