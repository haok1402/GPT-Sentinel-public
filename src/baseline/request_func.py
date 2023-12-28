import requests

zerogpt_headers = {
    "Content-Type": "application/json",
    "Origin": "https://www.zerogpt.com",
    "Referer": "https://www.zerogpt.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
}

zerogpt_url = "https://api.zerogpt.com/api/detect/detectText"

def zerogpt(uid: str, text: str) -> dict:
    try:
        response = requests.post(zerogpt_url, headers=zerogpt_headers, json={"input_text": text})
        return {'uid': uid, 'res': response.json()}, response.status_code
    except:
        return {'uid': uid, 'res': None}, 500


openai_url = "https://api.openai.com/v1/completions"

openai_json = {
    "max_tokens": 1,
    "temperature": 1,
    "top_p": 1,
    "n": 1,
    "logprobs": 5,
    "stop": "\n",
    "stream": False,
    "model": "model-detect-v2"
}

def openai(uid: str, text: str, header: dict) -> dict:
    try:
        openai_json['prompt'] = text + "<|disc_score|>"
        response = requests.post(openai_url, headers=header, json=openai_json)
        if response.status_code == 200: 
            return {'uid': uid, 'res': response.json()}, response.status_code
        else:
            return {'uid': uid, 'res': None}, response.status_code
    except Exception as e:
        return {'uid': uid, 'res': None}, 500
