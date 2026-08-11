"""Publish one rendered clip to YouTube via the poster service (per clip).

Usage: python scripts/publish_clip.py <clip_id> <job_id> [privacy]
Reads SEO from the miner DB, resolves the render output under
rendered/<job_id>/short_01.mp4 + thumbnail.jpg, POSTs to the poster
service, and syncs publish_status/url back into the miner DB.
"""
import json
import os
import subprocess
import sys

MINER_DB = "/data/content-miner.db"
POSTER = "http://127.0.0.1:8085/api/publish"
ROOT = r"D:/homelab/hermes-workspace/AI-Youtube-Shorts-Generator"

DB_Q = """
const{DatabaseSync}=require('node:sqlite');
const db=new DatabaseSync('%s');
const r=db.prepare('SELECT seo_title,seo_description,seo_tags FROM clips WHERE id=%s').get();
if(!r){console.log('NO_CLIP');process.exit(1);}
console.log(JSON.stringify({t:r.seo_title,d:r.seo_description||'',tags:JSON.parse(r.seo_tags||'[]')}));
"""

def db_js(code: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "content-miner", "node", "-e", code],
        capture_output=True, text=True, timeout=60,
    )
    lines = [l for l in out.stdout.splitlines() if "Warning" not in l and "trace" not in l]
    return lines[-1] if lines else ""

def main() -> int:
    clip_id, job_id = int(sys.argv[1]), sys.argv[2]
    privacy = sys.argv[3] if len(sys.argv) > 3 else "public"

    seo_raw = db_js(DB_Q % (MINER_DB, clip_id))
    if seo_raw == "NO_CLIP":
        print(f"clip {clip_id} not found"); return 1
    seo = json.loads(seo_raw)

    job_dir = os.path.join(ROOT, "rendered", job_id)
    video = os.path.join(job_dir, "short_01.mp4")
    thumb = os.path.join(job_dir, "thumbnail.jpg")
    if not os.path.exists(video):
        print(f"missing video: {video}"); return 1

    payload = {
        "clip_id": clip_id,
        "title": seo["t"],
        "description": seo["d"],
        "tags": seo["tags"],
        "file_url": video.replace("\\", "/"),
        "thumbnail_url": (thumb if os.path.exists(thumb) else "").replace("\\", "/"),
        "privacy": privacy,
    }
    with open(os.path.join(job_dir, "publish-payload.json"), "w") as f:
        json.dump(payload, f, indent=2)

    r = subprocess.run(
        ["curl", "-s", "-m", "300", "-X", "POST", POSTER,
         "-H", "Content-Type: application/json",
         "-d", "@" + os.path.join(job_dir, "publish-payload.json")],
        capture_output=True, text=True, timeout=360,
    )
    resp = json.loads(r.stdout or "{}")
    print(json.dumps(resp, indent=2))
    if resp.get("status") == "published" and resp.get("videoId"):
        db_js(
            f"const{{DatabaseSync}}=require('node:sqlite');"
            f"const db=new DatabaseSync('{MINER_DB}');"
            f"db.prepare(\"UPDATE clips SET publish_status='published',"
            f"publish_url='https://youtu.be/{resp['videoId']}',"
            f"published_at=datetime('now'),render_status='done' "
            f"WHERE id={clip_id}\").run(); console.log('db synced');"
        )
        print(f"OK published: https://youtu.be/{resp['videoId']}")
        return 0
    print("PUBLISH FAILED:", resp.get("error") or resp.get("detail"))
    return 1

if __name__ == "__main__":
    sys.exit(main())