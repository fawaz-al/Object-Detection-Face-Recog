"""
Lab 3 - Sistem Face Recognition Lokal dengan InsightFace.

Struktur folder yang diharapkan:
    face_recognition_demo/
        known_faces/
            NamaOrang1/*.jpg
            NamaOrang2/*.jpg
        test_images/*.jpg
        face_db.pkl   (auto-generated)
        lab3_recognize.py

Instalasi:
    pip install insightface onnxruntime opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from insightface.app import FaceAnalysis

# ============================================================
# KONFIGURASI
# ============================================================
KNOWN_FACES_DIR = Path("known_faces")
TEST_DIR = Path("test_images")
DB_PATH = "face_db.pkl"
THRESHOLD = 0.45
MODEL_NAME = "buffalo_l"

app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


# ============================================================
# 1. BUILD DATABASE
# ============================================================
def build_database():
    db = {}
    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        embeddings = []

        for img_path in sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Tidak dapat membaca {img_path}, dilewati.")
                continue
            faces = app.get(img)
            if not faces:
                print(f"[WARN] Tidak ada wajah ditemukan di {img_path.name}")
                continue
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            embeddings.append(face.embedding)

        if embeddings:
            mean_emb = np.stack(embeddings).mean(axis=0)
            db[person_name] = mean_emb / np.linalg.norm(mean_emb)
            print(f"Terdaftar: {person_name} ({len(embeddings)} foto)")
        else:
            print(f"[SKIP] Tidak ada wajah valid untuk {person_name}")

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"\nDatabase disimpan ke {DB_PATH} ({len(db)} identitas)")
    return db


def build_database_from_n_photos(person_dir: Path, n: int):
    """Untuk tugas FA2: enroll menggunakan hanya n foto pertama."""
    photos = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))[:n]
    embeddings = []
    for img_path in photos:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embeddings.append(face.embedding)
    if not embeddings:
        return None
    mean_emb = np.stack(embeddings).mean(axis=0)
    return mean_emb / np.linalg.norm(mean_emb)


# ============================================================
# 2. RECOGNIZE SINGLE IMAGE
# ============================================================
def recognize_image(img_path: str, db: dict, threshold: float = THRESHOLD):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Tidak dapat membaca {img_path}")
        return None, []

    faces = app.get(img)
    results = []

    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        query_emb = face.embedding / np.linalg.norm(face.embedding)

        best_name, best_sim = "Unknown", -1.0
        for name, ref_emb in db.items():
            sim = float(np.dot(query_emb, ref_emb))
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim < threshold:
            label = f"Unknown ({best_sim:.2f})"
            color = (0, 0, 220)
        else:
            label = f"{best_name} ({best_sim:.2f})"
            color = (0, 200, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        results.append((best_name, best_sim))

    return img, results


# ============================================================
# 3. LIVE RECOGNITION (WEBCAM)
# ============================================================
def live_recognition(db: dict):
    cap = cv2.VideoCapture(0)
    print("Webcam dimulai. Tekan 'q' untuk keluar, 'r' untuk rebuild DB.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            query_emb = face.embedding / np.linalg.norm(face.embedding)

            best_name, best_sim = "Unknown", -1.0
            for name, ref_emb in db.items():
                sim = float(np.dot(query_emb, ref_emb))
                if sim > best_sim:
                    best_sim, best_name = sim, name

            color = (0, 0, 220) if best_sim < THRESHOLD else (0, 200, 0)
            label = ("Unknown" if best_sim < THRESHOLD else best_name) + f" ({best_sim:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Face Recognition (tekan q untuk keluar)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            print("Membangun ulang database...")
            db = build_database()

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 4. TUGAS ANALISIS FA1: distribusi similarity genuine vs impostor
# ============================================================
def evaluate_threshold_sensitivity(db: dict, test_dir: Path = TEST_DIR):
    """
    Untuk setiap foto uji, cocokkan dengan seluruh database.
    - Jika nama folder foto == nama identitas terbaik -> genuine pair.
    - Jika berbeda -> impostor pair.
    Mengasumsikan test_images/<NamaOrang>/*.jpg berisi foto ground-truth berlabel.
    """
    genuine_scores, impostor_scores = [], []

    for person_dir in sorted(test_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        true_name = person_dir.name
        for img_path in sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            query_emb = face.embedding / np.linalg.norm(face.embedding)

            for name, ref_emb in db.items():
                sim = float(np.dot(query_emb, ref_emb))
                if name == true_name:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)

    print(f"Genuine pairs: {len(genuine_scores)} | Impostor pairs: {len(impostor_scores)}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(genuine_scores, bins=20, alpha=0.6, label="Genuine", color="green")
        plt.hist(impostor_scores, bins=20, alpha=0.6, label="Impostor", color="red")
        plt.xlabel("Cosine Similarity"); plt.ylabel("Frekuensi")
        plt.title("Distribusi Genuine vs Impostor Similarity")
        plt.legend(); plt.grid(True)
        plt.savefig("fa1_threshold_distribution.png", dpi=150)
        print("Plot disimpan: fa1_threshold_distribution.png")
    except ImportError:
        print("matplotlib tidak tersedia, lewati plotting.")

    return genuine_scores, impostor_scores


def roc_curve_manual(genuine_scores, impostor_scores, thresholds=None):
    """Tugas FA/Lab3 langkah 3: hitung TAR & FAR untuk beberapa threshold."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]
    results = []
    g = np.array(genuine_scores)
    im = np.array(impostor_scores)
    for t in thresholds:
        tar = (g >= t).mean() if len(g) else float("nan")
        far = (im >= t).mean() if len(im) else float("nan")
        results.append((t, tar, far))
        print(f"threshold={t:.2f}  TAR={tar:.3f}  FAR={far:.3f}")
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        print(f"Memuat database yang ada dari {DB_PATH}")
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        print(f"Dimuat {len(db)} identitas: {list(db.keys())}")
    else:
        print("Membangun database baru dari folder known_faces/...")
        db = build_database()

    print("\nPilih mode:")
    print("1 - Uji pada file gambar tunggal")
    print("2 - Pengenalan live webcam")
    print("3 - Rebuild database")
    print("4 - Evaluasi threshold (FA1) memakai test_images/<Nama>/*.jpg")
    mode = input("Masukkan pilihan: ").strip()

    if mode == "1":
        img_path = input("Masukkan path gambar: ").strip()
        result_img, results = recognize_image(img_path, db)
        if result_img is not None:
            print("Hasil:", results)
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif mode == "2":
        live_recognition(db)
    elif mode == "3":
        db = build_database()
    elif mode == "4":
        g, im = evaluate_threshold_sensitivity(db)
        roc_curve_manual(g, im)
    else:
        print("Pilihan tidak valid.")
