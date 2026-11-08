import json
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("set API Key")

def get_noun_list_from_caption(caption: str, retries: int = 3, timeout: int = 30):
    prompt = f"""
You are an NLP tool. From the following English caption, extract ONLY the nouns as a Python list of lowercase words.
Remove adjectives and other words.
Caption: "{caption}"
Return ONLY a valid JSON array, no extra text. Example: ["port", "buildings", "trees"]
"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = content.strip("`")
                if content.lstrip().startswith("json"):
                    content = content[content.find("\n")+1:]
                content = content.strip()

            noun_list = json.loads(content)
            if not isinstance(noun_list, list):
                noun_list = []
            return noun_list
        except Exception as e:
            print(f"[WARN] failure retry {attempt+1}/{retries} {e}")
            time.sleep(1)  
    return []

def process_item(idx, caption):
    nouns = get_noun_list_from_caption(caption)
    return idx, nouns

def main():
    with open("rsicd_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    max_workers = 16

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, item in enumerate(data):
            cap = item["caption"]
            future = executor.submit(process_item, idx, cap)
            futures.append(future)

        for future in as_completed(futures):
            idx, nouns = future.result()
            data[idx]["nouns"] = nouns
            if idx % 50 == 0:
                print(f"finish {idx}/{len(data)} nouns examples:{nouns}")

    with open("rsicd_train1.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("finished")

if __name__ == "__main__":
    main()
