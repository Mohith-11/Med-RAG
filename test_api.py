import requests

res = requests.post("http://127.0.0.1:8080/api/agent", json={"query": "What is the most common presenting symptom for head and neck cancers?"})
data = res.json()
print("Answer:", data.get('answer'))
print("\nSources:")
for s in data.get('sources', []):
    print(f"- {s['title']} (p.{s['page']}): {s['text'][:100]}")
