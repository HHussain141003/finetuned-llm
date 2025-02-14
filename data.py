import json
import re

with open("cleaned_json.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for entry in data:
    text = entry["content"]
    
    questions = re.findall(r"(?<=\n)(What|How|Why|When|Where|Can)[^\n?]+\?", text)
    
    for question in questions:
        formatted_data.append({
            "instruction": question.strip(),
            "response": text,
            "context": f"Source: {entry['url']}"
        })

with open("lora_dataset.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)
