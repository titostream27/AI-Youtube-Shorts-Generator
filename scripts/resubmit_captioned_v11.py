import sqlite3, json, time, urllib.request

c = sqlite3.connect(r'D:\homelab\hermes-workspace\content-miner\data\content-miner.db')
BASE = "http://127.0.0.1:8084"

CLIPS = [
    ("Ive926sC6mc", "https://www.youtube.com/watch?v=Ive926sC6mc", 2097.5, 2138.0, "Why Gen Z Craves Human Connection"),
    ("g2cQ2kD6lzs", "https://www.youtube.com/watch?v=g2cQ2kD6lzs", 0.0, 35.08, "It's Okay to Fail"),
    ("I6wCuvvaRPI", "https://www.youtube.com/watch?v=I6wCuvvaRPI", 638.7, 686.0, "Chloe Got Me the Bag"),
]

def get_cues(vid, start, end):
    row = c.execute("SELECT cues FROM transcripts WHERE video_id=?", (vid,)).fetchone()
    if not row:
        return []
    cues = json.loads(row[0])
    return [cu for cu in cues if cu.get("startSec") is not None and start - 0.8 <= cu.get("startSec") < end + 0.5]

def build_caption_plan(cues, clip_start, clip_end):
    trimmed = []
    for cu in cues:
        text = (cu.get("text") or "").strip()
        if not text:
            continue
        s = max(cu["startSec"], clip_start)
        e = min(cu.get("endSec") or cu["startSec"] + 2, clip_end)
        if s >= clip_end - 0.1:
            continue  # cue starts at/after the clip end -> skip
        e = min(max(e, s + 0.3), clip_end)
        if e <= s:
            continue
        trimmed.append({"start_sec": round(s, 3), "end_sec": round(e, 3), "text": text, "words": []})
    # Contract: cues must be ordered by start and must not overlap.
    trimmed.sort(key=lambda x: x["start_sec"])
    ordered = []
    last_end = clip_start
    for cu in trimmed:
        if cu["start_sec"] < last_end:
            cu["start_sec"] = last_end
            if cu["end_sec"] <= cu["start_sec"]:
                cu["end_sec"] = min(cu["start_sec"] + 0.3, clip_end)
        if cu["end_sec"] > clip_end:
            cu["end_sec"] = clip_end
        if cu["end_sec"] <= cu["start_sec"] or cu["start_sec"] >= clip_end:
            continue
        ordered.append(cu)
        last_end = cu["end_sec"]
    return {
        "language": "en",
        "highlight_terms": [],
        "cues": ordered,
    }

for idx, (vid, url, start, end, title) in enumerate(CLIPS, start=1):
    cues = get_cues(vid, start, end)
    print(f"{vid}: {len(cues)} cues in window")
    body = {
        "contract_version": "2.0",
        "request_id": f"v11-b2-caption-{vid}-{int(time.time())}",
        "episode_id": vid,
        "video_url": url,
        "mode": "final",
        "source_preferences": {"max_height": 720, "prefer_best_available": False},
        "clips": [{
            "clip_id": idx,
            "start_sec": start,
            "end_sec": end,
            "title": title,
            "narrative": {"main_topic": title, "ending_type": "CONCLUSION"},
            "layout_plan": {"preferred_layout": "auto"},
            "caption_plan": build_caption_plan(cues, start, end),
            "editing_events": [],
        }],
    }
    req = urllib.request.Request(BASE + "/api/render/async", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    if idx == 1:
        cues_json = body["clips"][0]["caption_plan"]["cues"]
        bad = [cu for cu in cues_json if cu["start_sec"] < start or cu["end_sec"] > end or cu["start_sec"] >= cu["end_sec"]]
        print(f"  debug first clip: cues={len(cues_json)} bad={bad}")
    with urllib.request.urlopen(req, timeout=30) as r:
        print("  ->", json.loads(r.read()))