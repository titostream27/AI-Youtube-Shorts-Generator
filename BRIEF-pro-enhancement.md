# BRIEF — Enhance AI YouTube Shorts Creator ke Level Profesional

**Tujuan:** Naikkan kualitas output dari "otomatis & fungsional" ke "terlihat dibuat editor profesional". Brief ini untuk AI agent yang akan mengerjakan improvement secara bertahap.

**Repo:** `D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator`
**Bahasa:** Python. Render lewat FastAPI service (port 8084), publish lewat poster service (port 8085).

---

## 0. Aturan Kerja untuk Agent

1. **Satu perubahan aman per iterasi**, lalu render preview (540×960) untuk verifikasi visual sebelum lanjut. Jangan langsung final/publish.
2. **Jangan republish** video yang sudah live tanpa persetujuan eksplisit pemilik.
3. **Verifikasi objektif**: setiap klaim "lebih stabil / lebih tajam / lebih rapi" harus didukung metrik (jitter px, QC score, ffprobe) atau frame sampling — bukan asumsi.
4. **Jangan tampilkan** kredensial (`.env`, token YouTube OAuth, API key).
5. Test kontrak wajib hijau sebelum commit: `python -m pytest test_render_contract.py -q` (11 test).
6. Cache reframe caption-free ada di `rendered/cache/` — hapus cache klip terkait setelah mengubah `_reframe_vertical`, kalau tidak perubahan tidak akan terlihat.

---

## 1. Peta Sistem Saat Ini (verified)

Pipeline: `shorts_generator/pipeline.py` → `generate_shorts()`
- **Seleksi klip:** `shorts_generator/highlights.py` — LLM memilih highlight dari transcript (Whisper).
- **Reframe vertikal + tracking + split-panel:** `shorts_generator/local/clipper.py` `_reframe_vertical()`.
- **Layout pick:** `visual_effects.py` `choose_layout()` (auto → blur_background / split / dll).
- **Color grade:** `visual_effects.py` `build_color_filter()`.
- **Emphasis events (caption pop):** `visual_effects.py` `build_emphasis_events()`.
- **Audio master:** `audio_master.py` `build_audio_chain()` / `master_audio()` / `trim_pauses()`.
- **Quality gate:** `quality_gate.py` `run_quality_checks()` — skor 100 dikurangi penalti (upscale, black/frozen frame, av-sync, loudness, dst). Min lulus = `QC_MIN_SCORE`.
- **Publish + SEO + thumbnail:** `poster_service.py` — upload YouTube (title≤100, desc≤5000, tags), set custom thumbnail.

Yang sudah bagus: tracking anti-shake (EMA + dead-zone per panel), speaker lock, QC gate, caption face-avoidance, custom thumbnail.

---

## 2. Ide Improvement (prioritas: dampak visual tertinggi → terendah)

### TIER 1 — Paling kelihatan "pro" (kerjakan dulu)

**1.1 Caption / subtitle styling profesional**
- Saat ini caption polos + face-avoidance. Upgrade ke gaya shorts modern:
  - Active-word highlight (karaoke): kata yang sedang diucapkan diberi warna/scale beda.
  - Font tebal dengan stroke + drop shadow konsisten (readability di layar HP).
  - Maks 3–4 kata per baris, auto-break di jeda natural (pakai word timestamps Whisper yang sudah ada).
  - Safe-zone: jangan pernah timpa wajah (logic sudah ada di `_LAST_FACE_TRACKS`) DAN jangan timpa panel speaker aktif (lihat isu panel dinamis di §2.4).
- **Verifikasi:** frame sampling di 3 titik, cek keterbacaan + tidak nutup wajah.

**1.2 Hook 3 detik pertama (retention)**
- 3 detik pertama penentu retention. Tambahkan:
  - Auto-generate hook text overlay dari `seoTitle`/highlight (sudah ada `hook` di tracking_stats).
  - Opsi: freeze-frame + zoom-in kecil di kata kunci pembuka.
- **Verifikasi:** cek hook muncul 0–3s, tidak menutup wajah.

**1.3 B-roll / cutaway ringan (opsional, tinggi effort)**
- Untuk momen "telling" tanpa aksi visual, sisipkan zoom-punch (Ken Burns) halus alih-alih static frame. Reuse EMA state yang ada.

### TIER 2 — Polish yang dirasakan penonton

**2.1 Audio profesional**
- `audio_master.build_audio_chain()`: pastikan ada loudness normalization ke −14 LUFS (standar YouTube), de-ess ringan, dan noise gate.
- QC sudah ukur `audio_lufs`/`true_peak` (saat ini null di beberapa render) — pastikan terisi & masuk gate.
- **Verifikasi:** `ffprobe`/`loudnorm` print LUFS after; target −14 ±1.

**2.2 Dynamic pacing / pause trimming**
- `audio_master.trim_pauses()` sudah ada — pastikan aktif di pipeline agar dead-air >0.4s dipangkas. Naikkan "energy" klip.
- **Verifikasi:** durasi sebelum/sesudah, cek tidak memotong kata.

**2.3 Color grade konsisten**
- `build_color_filter()`: satu LUT/curve ringan (contrast + saturation + sedikit warmth) agar brand look konsisten antar-klip. Hindari over-saturate.

**2.4 Panel dinamis di split-blur (ISU AKTIF milik owner)**
- Masalah: speaker aktif kadang ke-lock di panel BAWAH, tertutup subtitle.
- Fix: di `_reframe_vertical` blok split-blur (sekitar baris 1124–1220), antara 2 track yang di-lock (`spk` = `speaker_track_id`, `second` = `blur_second_id`), tempatkan yang **sedang bicara** (aktivitas mulut tertinggi) di panel ATAS.
- Wajib: hold-time (mis. ≥0.8s) + cross-fade saat swap, biar tidak lompat-lompat tiap ganti pembicara.
- **Verifikasi:** render preview klip 364, cek speaker selalu di atas & swap mulus.

### TIER 3 — Distribusi & metadata (bikin "channel" terlihat pro)

**3.1 SEO otomatis lebih kuat**
- `poster_service.py`: generate title (hook-driven, ≤100), description (ringkasan + timestamp + CTA + hashtag relevan), tags dari topik klip via LLM. Konsisten template per channel.

**3.2 Thumbnail pintar**
- Custom thumbnail sudah didukung. Tambah: pilih frame dengan wajah ekspresif (pakai tracking activity), overlay judul besar 2–4 kata, kontras tinggi.

**3.3 Konsistensi brand**
- Intro/outro sting 0.5s opsional, watermark handle channel di pojok (non-intrusive), font family tetap.

---

## 3. Urutan Eksekusi yang Disarankan

1. **§2.4 panel dinamis** (isu aktif, cepat, dampak jelas) → preview → owner approve.
2. **§1.1 caption styling** (dampak visual terbesar) → preview 3 frame.
3. **§2.1 audio −14 LUFS** (mudah diverifikasi objektif).
4. **§1.2 hook 3 detik**.
5. **§2.2 pause trimming** + **§2.3 color**.
6. **§3.x SEO/thumbnail/brand**.

Setiap langkah: patch → `pytest` hijau → render preview → verifikasi metrik/visual → laporkan ke owner → tunggu approve untuk final/publish.

---

## 4. Definition of Done per Item

- Kode: test kontrak hijau, tidak ada regresi QC (skor tetap ≥ ambang).
- Visual: frame sampling membuktikan klaim (bukan asumsi).
- Audio: LUFS terukur mendekati −14.
- Tidak ada kredensial bocor.
- Owner meng-approve sebelum final render + publish.

---

## 5. Catatan Realitas (jangan over-promise)

- Ini **peningkatan bertahap**, bukan rewrite. Reuse state machine tracking & QC yang sudah matang.
- Fitur "AI B-roll", "auto music sync", "multi-language dub" = effort besar; tandai sebagai backlog, jangan campur ke iterasi pendek.
- Semua klaim performa perlu diverifikasi di mesin ini (Windows, RX 6700 XT, ffmpeg lokal). Jangan asumsi hasil dari dokumentasi.
