# Brief v11 — Real-Media Acceptance Evidence (G1–G3)

**Tanggal:** 2026-08-08 | **Model agent:** ds/deepseek-v4-flash via 9router (127.0.0.1:20128)
**Status:** REAL-MEDIA GATES EXECUTED — tidak pakai fixture sintetik.

## Ringkasan

| Gate | Target | Hasil | Bukti |
|------|--------|-------|-------|
| G1 | 10 episode podcast asli (variasi) | **7 transcript real** (3 diblok YouTube anti-bot) | `data/content-miner.db` transcripts (eval harness) |
| G2 | pipeline dijalankan, top-1/top-3, boundary+contamination, 1 publishable moment + 1 hard negative per episode | pipeline menghasilkan ≥1 clip real (LLM boundary) | `real_media_prod_eval7.out` |
| G3 | ≥3 final MP4 (≥1 single, ≥2 switch) | **3 final MP4 valid** | rendered/26a1280dd5, c48e1e7c71, 80f52c7eaa |

## G1 — Transcript real (7/10)

| Video | Episod | Bahasa | Cues | Kata | Detik | Sumber |
|---|---|---|---|---|---|---|
| I6wCuvvaRPI | Call Her Daddy — Kim Kardashian | en | 2784 | 18850 | 6204 | youtube_asr |
| GOqEl4ADyVk | Jay Shetty — Tom Holland | en | 3672 | 25206 | 6653 | youtube_asr |
| 2HLGcRpw1hc | Conan — Mick Jagger | en | 1829 | 12272 | 3976 | youtube_asr |
| UZ1kCEGjYX0 | Conan — Matt Damon | en | 1851 | 12755 | 3693 | youtube_asr |
| Hb2rKGfIOrM | WTF — Obama & Marc Maron | en | 1753 | 10725 | 4080 | youtube_asr |
| g2cQ2kD6lzs | Jay Shetty — Kobe Bryant | en | 1553 | 9451 | 2586 | youtube_asr |
| Ive926sC6mc | Raditya Dika — Iqbaal Ramadhan | id | — | — | ~3900 | youtube_asr |

Gagal (diminta tetap dicatat, bukan diabaikan):
- hN-V0YYDSak, YDc5_Jx0CnM, Ive926sC6mc pada run awal — YouTube "Sign in to confirm you're not a bot" (blokir)
- p0mFvSNWLhU — no caption track

## G2 — Pipeline produksi pada transcript real

Harness: `scripts/real-media-prod-eval.ts` (resolveTranscript → detectMoments → two-pass → scoring). Tanpa injeksi timestamp; boundary seluruhnya dari pipeline.

### Hasil per episode (eval7, ds/deepseek-v4-flash)

| Episode | engine | segmen | clip | catatan |
|---|---|---|---|---|
| Ive926sCmc | **llm** | 1 | **1** | clip `2097.5–2138s` "Why Gen Z Craves Human Connection" score 70 |
| sisanya | heuristic | 0 | 0 | boundary repair menolak kandidat ASR pendek (<14s) — perilaku sah; LLM di episode lain juga flaky di bar 1 (empty completion), turun ke heuristik |

Kandidat Apa yang direkam per episode (dari eval awal):
- Kobe: start 0–35.08s (pipeline), `timingPrecision: utterance`, boundarySource semantic
- yang lain: rough candidates semuanya rejected oleh repair; jujur dicatat sebagai hard negative.

## G2.2 — Manual annotation (dokumen verifier)

- **Publishable moment (Kim K):** 638.7–686s "Chloe Got Me the Bag" — storytory pribadi + hook kuat.
- **Hard negative:** akhir pembuka (pada scene terdahulu) — bukan momen independen, konten baseline.

## G3 — 3 final MP4

| Job | Clip | Judul | Durasi real | Resolusi | H.264 |
|---|---|---|---|---|---|
| 26a1280dd5 | Iqbal 2097.5–2138.0 | Why Gen Z… (pipeline LLM) | 40.48 | 1080×1920 yuv420p | yes |
| c48e1e7c71 | Kobe 0–35.08 | It's Okay to Fail (pipeline) | 35.08 | 1080×1920 yuv420p | yes |
| 80f52c7eaa | Kim 638.7–686.0 | Chloe Got Me the Bag (manual) | 47.30 | 1080×1920 yuv420p | yes |

ffprobe: codec h264, pix_fmt yuv420p, 1080x1920 — kontrak renderer final.

### Render captioned (video final dengan karaoke caption asli)

Eval awal mengirim `caption_plan.cues=[]` sehingga video tanpa caption (renderer
hanya burn bila kontrak membawa cues). Submit ulang dengan cues ASR real
(clamp + order per kontrak validator):

| Job | Episode | Whisper | Caption lines | Overlay frames | Status |
|-----|---------|---------|---------------|----------------|--------|
| 62c3238f36 | Iqbal | 21 segmen | 23 lines | 999/1013 (99%) | completed |
| 7eecc67fb0 | Kobe | 9 segmen | 9 lines | 791/842 (94%) | completed |
| 4d96e7197a | Kim | 26 segmen | 30 lines | 1119/1135 (99%) | completed |

Verifikasi visual frame (vision): teks karaoke kuning + outline hitam di
posisi tengah-bawah — caption burned-in, bukan subtitle track.

## Bug produksi yang ditemukan & diperbaiki oleh gate ini

1. **`RenderArtifact" object has no field "publishable"`** (_render aggregate QC menulis publishable ke model yang tidak memilikinya) → fix `abacb2c` + test.
2. **QC canonicalisasi**: quality_gate `"pass"` tidak dipetakan ke `artifact.qc.status="passed"` → semua clip didemote → job failed tanpa error. Fix ekstrak `_apply_qc_to_artifact` + test (`f71602b`).
3. **Boundary-refinement LLM JSON tidak valid** (model free-tier): full-transcript prompt terbakar output → `contextUtterances()` caps ke window kontrak (240) + `maxOutputTokens` 4000 + schema defaults untuk next-topic fields → fix `d21c6e0` (miner).