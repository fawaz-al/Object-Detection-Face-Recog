# Object-Detection-Face-Recog

Repositori ini berisi implementasi *pipeline* *Computer Vision* yang menggabungkan deteksi objek secara umum dan pengenalan wajah tingkat lanjut. Proyek ini dibangun menggunakan model *state-of-the-art* untuk memastikan akurasi dan efisiensi yang tinggi dalam mendeteksi serta mengenali entitas di dalam gambar atau *video stream*.

## 🚀 Fitur Utama
* **Deteksi Objek (Object Detection):** Mendeteksi berbagai macam objek dalam satu *frame* dengan akurasi tinggi menggunakan Faster R-CNN.
* **Pengenalan Wajah (Face Recognition):** Melacak, mengekstraksi fitur, dan mengenali identitas wajah menggunakan InsightFace (`buffalo_l`).
* **Integrasi Seamless:** Proses *end-to-end* dari *input* gambar mentah hingga *output* berupa *bounding box* dan label nama/objek.

## 🧠 Model yang Digunakan
1. **Faster R-CNN:** Arsitektur *deep learning* yang sangat andal untuk tugas deteksi objek, mampu memberikan keseimbangan yang baik antara kecepatan dan akurasi komputasi.
2. **InsightFace (`buffalo_l`):** Model pengenalan wajah berkinerja tinggi (*Large model*) yang unggul dalam ekstraksi fitur wajah (*face embedding*), deteksi *landmark*, dan perbandingan identitas.

## 🛠️ Teknologi & Library (Tech Stack)
* Python 3.x
* PyTorch / Torchvision
* InsightFace
* OpenCV (cv2)
* NumPy

## ⚙️ Instalasi (Installation)
Ikuti langkah-langkah berikut untuk menjalankan proyek ini di mesin lokal Anda:

1. *Clone* repositori ini:
   ```bash
   git clone [https://github.com/fawaz-al/Object-Detection-Face-Recog.git](https://github.com/fawaz-al/Object-Detection-Face-Recog.git)
   cd Object-Detection-Face-Recog

2. Buat virtual environment (Sangat Disarankan):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
3. Install semua dependesni yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
## 💻 Cara Penggunaan (Usage)
Contoh penggunaan singkat untuk memproses sebuah gambar:
    ```python
    import cv2
    # Import modul pipeline buatan Anda di sini
    # from src.pipeline import run_detection
    
    image_path = 'data/test_image.jpg'
    image = cv2.imread(image_path)
    
    # Jalankan deteksi
    # result_image = run_detection(image)
    
    # cv2.imshow('Result', result_image)
    # cv2.waitKey(0)

## Struktur Direktori (Folder Structure)
    ```text
    Object-Detection-Face-Recog/
    ├── data/               # Folder untuk menyimpan dataset atau gambar tes
    ├── models/             # Folder untuk menyimpan bobot model (weights)
    ├── src/                # Source code utama (modul deteksi)
    ├── main.py             # Script utama untuk menjalankan program
    ├── requirements.txt    # Daftar dependensi library Python
    └── README.md           # Dokumentasi proyek
    ```
