import json
with open('data/outputs/test_clip_result.json', encoding='utf-8') as f:
    data = json.load(f)

for s in data:
    sid = s['sentence_id']
    speaker = s['speaker']
    text = s['text'][:40]
    print(f'sentence {sid} | {speaker} | {text}')
