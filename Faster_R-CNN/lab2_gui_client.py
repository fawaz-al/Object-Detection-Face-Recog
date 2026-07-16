"""
Lab 2 - Frontend GUI (Tkinter) untuk layanan deteksi.
Pengguna memilih gambar (atau drag-and-drop jika tkinterdnd2 tersedia),
lalu gambar dikirim ke backend FastAPI (/detect) dan hasilnya digambar
sebagai bounding box di atas gambar.

Jalankan dengan:
    pip install requests pillow
    (opsional, untuk drag-and-drop) pip install tkinterdnd2
    python lab2_gui_client.py

Pastikan backend sudah berjalan:
    uvicorn lab2_fastapi_backend:app --reload --port 8000
"""

import io
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

BACKEND_URL = "http://127.0.0.1:8000/detect"

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    BASE_CLASS = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:
    BASE_CLASS = tk.Tk
    DND_AVAILABLE = False


COLORS = {"person": (0, 200, 0), "bicycle": (220, 120, 0)}


class DetectionApp(BASE_CLASS):
    def __init__(self):
        super().__init__()
        self.title("Layanan Deteksi Real-time - Lab 2")
        self.geometry("700x600")

        self.label_info = tk.Label(
            self, text="Klik 'Pilih Gambar' atau drag-and-drop file gambar ke sini",
            font=("Arial", 12)
        )
        self.label_info.pack(pady=10)

        self.canvas = tk.Label(self, bg="#dddddd")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Pilih Gambar", command=self.choose_file).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Keluar", command=self.destroy).pack(side="left", padx=5)

        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.label_info.config(
                text="Klik 'Pilih Gambar' untuk mengunggah "
                     "(install tkinterdnd2 untuk drag-and-drop)"
            )

    def choose_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.process_image(path)

    def on_drop(self, event):
        path = event.data.strip("{}")
        self.process_image(path)

    def process_image(self, path):
        try:
            with open(path, "rb") as f:
                files = {"file": (path, f, "image/jpeg")}
                resp = requests.post(BACKEND_URL, files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menghubungi backend:\n{e}")
            return

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for det in data["detections"]:
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(det["class_name"], (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{det['class_name']} {det['score']:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill=color)

        img.thumbnail((650, 500))
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(image=tk_img)
        self.canvas.image = tk_img  # keep reference
        self.label_info.config(text=f"Ditemukan {len(data['detections'])} objek")


if __name__ == "__main__":
    app = DetectionApp()
    app.mainloop()
