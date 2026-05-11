import json, re

with open("shl_raw.json", "rb") as f:
    raw = f.read()

cleaned = re.sub(rb'[\x00-\x1f\x7f](?<![\\n\\t\\r])', b' ', raw)
data = json.loads(cleaned, strict=False)

KEY_MAP = {
    "Knowledge & Skills":             "K",
    "Ability & Aptitude":             "A",
    "Personality & Behavior":         "P",
    "Biodata & Situational Judgment": "S",
    "Assessment Exercises":           "E",
    "Competencies":                   "C",
    "Development & 360":              "C",
}

items = []
for r in data:
    if r.get("status") != "ok":
        continue
    name = r.get("name", "").strip()
    url  = r.get("link", "").strip()
    if not name or not url:
        continue
    keys = r.get("keys", [])
    test_type = next((KEY_MAP[k] for k in keys if k in KEY_MAP), "C")
    desc = " | ".join(filter(None, [
        r.get("description", ""),
        "Job levels: " + r["job_levels_raw"] if r.get("job_levels_raw") else "",
        "Duration: "   + r["duration"]        if r.get("duration")        else "",
        "Remote: "     + r["remote"]          if r.get("remote")          else "",
        "Adaptive: "   + r["adaptive"]        if r.get("adaptive")        else "",
    ]))
    items.append({
        "name":        name,
        "url":         url,
        "description": desc,
        "test_type":   test_type,
        "job_levels":  r.get("job_levels", []),
        "duration":    r.get("duration", ""),
        "remote":      r.get("remote", ""),
        "adaptive":    r.get("adaptive", ""),
        "keys":        keys,
    })

with open("catalog.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

print(f"{len(items)} items written to catalog.json")
types = {}
for i in items:
    types[i["test_type"]] = types.get(i["test_type"], 0) + 1
print("Test type breakdown:", types)
