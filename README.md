---
title: Vehicle and Accident Detection
emoji: 🚗
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
---

# Vehicle and Accident Detection 🚗

Aplikasi deteksi kendaraan dan kecelakaan lalu lintas menggunakan **YOLOv11m** yang dilatih dengan augmentasi 3x.

## Fitur
- Upload video (MP4, AVI, MOV, MKV)
- Definisikan area deteksi dengan polygon (preset atau custom)
- Deteksi kendaraan: `bus`, `car`, `motorcycle`, `truck`
- Deteksi kecelakaan: berbagai kombinasi tabrakan antar kendaraan
- Statistik real-time & snapshot kecelakaan
- Download hasil (video + Excel) dalam format ZIP

## Model
- **YOLOv11m** — dilatih dengan augmentasi 3x (`Augmen3x-Yolov11m.pt`)

## Cara Pakai
1. Upload video di sidebar
2. Pilih atau gambar area deteksi
3. Klik **Start/Restart** untuk mulai deteksi
4. Download hasil setelah proses selesai
