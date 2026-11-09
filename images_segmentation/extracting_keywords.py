import argparse
import json
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

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
    parser = argparse.ArgumentParser(description="Add a 'nouns' field to each annotation with DeepSeek.")
    parser.add_argument("--input", required=True, help="Annotation JSON, e.g. data/finetune/rsicd_train.json")
    parser.add_argument("--output", required=True, help="Output JSON with nouns, e.g. data/finetune/rsicd_train1.json")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent API requests")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError("set the DEEPSEEK_API_KEY environment variable")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, item in enumerate(data):
            cap = item["caption"]
            future = executor.submit(process_item, idx, cap)
            futures.append(future)

        for future in as_completed(futures):
            idx, nouns = future.result()
            data[idx]["nouns"] = nouns
            if idx % 50 == 0:
                print(f"finish {idx}/{len(data)} nouns examples:{nouns}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"finished, written to {args.output}")

if __name__ == "__main__":
    main()
