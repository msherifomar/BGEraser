"""
BGEraser  v1.0.0
Courtesy of Dr. M S Omar  BDS  MSc
For Doers, Researchers and Innovators
===========================================================================
Offline AI background removal for research presentations, clinical imagery,
and educational materials.

Built on rembg with five selectable segmentation backbones and
alpha-matting controls for fine edges (hair, fabric, enamel margins).

HOW TO RUN:
    python BGEraser.py

REQUIRES:
    pip install rembg Pillow numpy onnxruntime
    pip install rembg[gpu]   # optional, if CUDA is available

First run downloads the selected model (~170 MB).
"""

import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk
import threading
import os
import sys
from pathlib import Path
import io

# Dependency guard
MISSING = []
try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
except ImportError:
    MISSING.append("Pillow")
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    from rembg import remove, new_session
except ImportError:
    MISSING.append("rembg")

if MISSING:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror(
        "Missing packages",
        f"Please install:\n  pip install {' '.join(MISSING)}\n\nthen restart."
    )
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────
# THEME
# ────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0F0F13",
    "panel":     "#16161D",
    "card":      "#1E1E28",
    "border":    "#2A2A38",
    "accent":    "#7C6AF7",
    "accent2":   "#F76A8C",
    "green":     "#3DFFA0",
    "text":      "#E8E8F0",
    "muted":     "#6B6B85",
    "white":     "#FFFFFF",
    "hover":     "#252535",
}

FONTS = {
    "title":   ("Segoe UI", 18, "bold"),
    "sub":     ("Segoe UI", 11, "bold"),
    "body":    ("Segoe UI", 10),
    "small":   ("Segoe UI", 9),
    "mono":    ("Consolas", 9),
    "badge":   ("Segoe UI", 8, "bold"),
}

# ────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ────────────────────────────────────────────────────────────────────────────
MODELS = {
    "BiRefNet - Ultra Precision (Hair)": "birefnet-general",
    "ISNet - Fine Details":              "isnet-general-use",
    "U2Net - Fast and Balanced":         "u2net",
    "U2Net Human - Portraits":           "u2net_human_seg",
    "SILUETA - Tight Mask":              "silueta",
}

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def make_checkerboard(w, h, size=16):
    img = Image.new("RGB", (w, h), "#2A2A38")
    draw = ImageDraw.Draw(img)
    for row in range(h // size + 1):
        for col in range(w // size + 1):
            if (row + col) % 2:
                x0, y0 = col * size, row * size
                draw.rectangle([x0, y0, x0+size, y0+size], fill="#1E1E28")
    return img

def fit_image(img, max_w, max_h):
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img

def composite_on_color(fg, bg_color):
    """Composite RGBA image onto solid colour background."""
    bg = Image.new("RGBA", fg.size, bg_color + (255,))
    return Image.alpha_composite(bg, fg.convert("RGBA")).convert("RGB")

# ────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────────────────────────────
class BGEraserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BGEraser  -  Offline Background Removal")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1280x800")
        self.root.minsize(960, 640)

        # state
        self.src_image = None
        self.result_image = None
        self.src_path = ""
        self.session = None
        self.session_name = ""
        self.bg_color = (255, 255, 255)
        self.bg_mode = tk.StringVar(value="transparent")
        self.alpha_matting = tk.BooleanVar(value=True)
        self.fg_thresh = tk.IntVar(value=240)
        self.bg_thresh = tk.IntVar(value=10)
        self.erode_size = tk.IntVar(value=10)
        self.model_var = tk.StringVar(value=list(MODELS.keys())[0])
        self.processing = False

        self._build_ui()
        self._update_bg_controls()

    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self.root, bg=C["panel"], height=60)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="BGERASER",
                 font=FONTS["title"], bg=C["panel"], fg=C["accent"]).pack(side="left", padx=20, pady=10)
        tk.Label(topbar, text="Offline  -  Privacy-Preserving  -  Presentation-Ready",
                 font=FONTS["small"], bg=C["panel"], fg=C["muted"]).pack(side="left", pady=14)

        badge = tk.Label(topbar, text=" v1.0 ", font=FONTS["badge"],
                         bg=C["accent"], fg=C["white"], padx=6, pady=2)
        badge.pack(side="right", padx=20, pady=16)

        # Main layout
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=12, pady=12)

        self.sidebar = tk.Frame(main, bg=C["panel"], width=280)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar.pack_propagate(False)

        preview_frame = tk.Frame(main, bg=C["bg"])
        preview_frame.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_preview(preview_frame)

        # Status bar
        self.statusbar = tk.Frame(self.root, bg=C["border"], height=28)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)
        self.status_lbl = tk.Label(self.statusbar, text="Ready  -  Import an image to begin",
                                   font=FONTS["mono"], bg=C["border"], fg=C["muted"], anchor="w")
        self.status_lbl.pack(side="left", padx=12, pady=4)

        self.progress = ttk.Progressbar(self.statusbar, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=12, pady=5)

    def _build_sidebar(self):
        sb = self.sidebar

        def section(label):
            f = tk.Frame(sb, bg=C["border"], height=1)
            f.pack(fill="x", padx=10, pady=(10, 0))
            tk.Label(sb, text=f"  {label}", font=FONTS["badge"],
                     bg=C["panel"], fg=C["muted"], anchor="w").pack(fill="x", padx=10)

        def card(parent=None):
            p = parent or sb
            f = tk.Frame(p, bg=C["card"], padx=10, pady=8)
            f.pack(fill="x", padx=10, pady=4)
            return f

        # IMPORT
        section("1. IMPORT")
        c = card()
        self._btn(c, "Open Image", self._open_image, accent=True).pack(fill="x")
        self.file_lbl = tk.Label(c, text="No file selected", font=FONTS["small"],
                                 bg=C["card"], fg=C["muted"], wraplength=220, anchor="w")
        self.file_lbl.pack(fill="x", pady=(6,0))

        # AI MODEL
        section("2. AI MODEL")
        c = card()
        tk.Label(c, text="Segmentation engine:", font=FONTS["small"],
                 bg=C["card"], fg=C["muted"]).pack(anchor="w")
        mdl_menu = tk.OptionMenu(c, self.model_var, *MODELS.keys())
        mdl_menu.config(bg=C["hover"], fg=C["text"], font=FONTS["small"],
                        activebackground=C["accent"], activeforeground=C["white"],
                        relief="flat", bd=0, highlightthickness=0)
        mdl_menu["menu"].config(bg=C["hover"], fg=C["text"], font=FONTS["small"],
                                 activebackground=C["accent"], activeforeground=C["white"])
        mdl_menu.pack(fill="x", pady=(4,0))

        am_frame = tk.Frame(c, bg=C["card"])
        am_frame.pack(fill="x", pady=(8,0))
        self._toggle(am_frame, "Alpha matting  (best for hair)", self.alpha_matting).pack(side="left")

        self.matting_frame = tk.Frame(c, bg=C["card"])
        self.matting_frame.pack(fill="x")

        def slider_row(parent, label, var, lo, hi):
            f = tk.Frame(parent, bg=C["card"])
            f.pack(fill="x", pady=2)
            tk.Label(f, text=label, font=FONTS["small"], bg=C["card"],
                     fg=C["muted"], width=14, anchor="w").pack(side="left")
            tk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var,
                     bg=C["card"], fg=C["text"], troughcolor=C["border"],
                     highlightthickness=0, relief="flat", sliderrelief="flat",
                     font=FONTS["small"], length=110).pack(side="left")

        slider_row(self.matting_frame, "FG threshold", self.fg_thresh, 180, 255)
        slider_row(self.matting_frame, "BG threshold", self.bg_thresh,  1,  50)
        slider_row(self.matting_frame, "Erode size",   self.erode_size, 2,  30)
        self.alpha_matting.trace_add("write", lambda *_: self._update_matting_sliders())
        self._update_matting_sliders()

        # OUTPUT BACKGROUND
        section("3. OUTPUT BACKGROUND")
        c = card()
        modes = [("Transparent  (PNG)",  "transparent"),
                 ("Solid colour  (PNG/JPG)", "color")]
        for lbl, val in modes:
            tk.Radiobutton(c, text=lbl, variable=self.bg_mode, value=val,
                           command=self._update_bg_controls,
                           bg=C["card"], fg=C["text"], selectcolor=C["accent"],
                           activebackground=C["card"], activeforeground=C["accent"],
                           font=FONTS["small"]).pack(anchor="w", pady=1)

        self.color_frame = tk.Frame(c, bg=C["card"])
        self.color_frame.pack(fill="x", pady=(4,0))
        tk.Label(self.color_frame, text="Background colour:", font=FONTS["small"],
                 bg=C["card"], fg=C["muted"]).pack(side="left")
        self.color_swatch = tk.Label(self.color_frame, text="       ", bg="#FFFFFF",
                                     relief="flat", cursor="hand2")
        self.color_swatch.pack(side="left", padx=6)
        self.color_swatch.bind("<Button-1>", self._pick_color)
        tk.Button(self.color_frame, text="Pick", font=FONTS["small"],
                  bg=C["hover"], fg=C["text"], relief="flat",
                  command=self._pick_color, cursor="hand2").pack(side="left")

        pf = tk.Frame(c, bg=C["card"])
        pf.pack(fill="x", pady=(4,0))
        tk.Label(pf, text="Presets:", font=FONTS["small"],
                 bg=C["card"], fg=C["muted"]).pack(side="left")
        for name, color in [("White","#FFFFFF"),("Black","#000000"),
                             ("Grey","#808080"),("Blue","#1E3A5F"),
                             ("Cream","#FFF8F0")]:
            btn = tk.Label(pf, text=" ", bg=color, width=2, cursor="hand2",
                           relief="raised")
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, col=color: self._set_color(col))

        # PROCESS
        section("4. PROCESS")
        c = card()
        self.run_btn = self._btn(c, "Remove Background", self._start_removal, accent=True)
        self.run_btn.pack(fill="x")
        self.run_btn.config(state="disabled")

        # EXPORT
        section("5. EXPORT")
        c = card()
        ef = tk.Frame(c, bg=C["card"])
        ef.pack(fill="x")
        self.save_png_btn = self._btn(ef, "Save PNG", self._save_png)
        self.save_png_btn.pack(side="left", fill="x", expand=True, padx=(0,3))
        self.save_png_btn.config(state="disabled")
        self.save_jpg_btn = self._btn(ef, "Save JPG", self._save_jpg)
        self.save_jpg_btn.pack(side="left", fill="x", expand=True)
        self.save_jpg_btn.config(state="disabled")

        tk.Frame(sb, bg=C["panel"]).pack(fill="both", expand=True)

        tip = tk.Label(sb, text="Tip: BiRefNet is best for portraits\nand fine hair strands",
                       font=FONTS["small"], bg=C["panel"], fg=C["muted"],
                       justify="center")
        tip.pack(pady=10)

    def _build_preview(self, parent):
        tab_bar = tk.Frame(parent, bg=C["bg"])
        tab_bar.pack(fill="x")

        self.view_mode = tk.StringVar(value="split")
        for lbl, val in [("Original", "original"), ("Split", "split"), ("Result", "result")]:
            rb = tk.Radiobutton(tab_bar, text=lbl, variable=self.view_mode, value=val,
                                command=self._refresh_preview,
                                bg=C["bg"], fg=C["muted"], selectcolor=C["accent"],
                                activebackground=C["bg"], activeforeground=C["accent"],
                                indicatoron=False, relief="flat",
                                font=FONTS["sub"], padx=14, pady=6,
                                cursor="hand2")
            rb.pack(side="left", padx=2)

        canvas_bg = tk.Frame(parent, bg=C["card"], bd=0)
        canvas_bg.pack(fill="both", expand=True, pady=(6,0))

        self.canvas = tk.Canvas(canvas_bg, bg=C["card"], highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh_preview())

        self.drop_overlay = tk.Label(
            self.canvas,
            text="Drop an image here\nor use  'Open Image'\n\nSupports PNG, JPG, WEBP, BMP, TIFF",
            font=("Segoe UI", 13), bg=C["card"], fg=C["muted"],
            justify="center"
        )
        self.drop_overlay.place(relx=0.5, rely=0.5, anchor="center")

    def _btn(self, parent, text, cmd, accent=False):
        bg  = C["accent"] if accent else C["hover"]
        fg  = C["white"]
        abg = C["accent2"] if accent else C["border"]
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
                        relief="flat", bd=0, padx=10, pady=7,
                        font=FONTS["body"], cursor="hand2")
        btn.bind("<Enter>", lambda e: btn.config(bg=abg))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def _toggle(self, parent, text, var):
        return tk.Checkbutton(parent, text=text, variable=var,
                              bg=parent["bg"], fg=C["text"],
                              selectcolor=C["accent"],
                              activebackground=parent["bg"], activeforeground=C["accent"],
                              font=FONTS["small"])

    def _update_bg_controls(self):
        state = "normal" if self.bg_mode.get() == "color" else "disabled"
        for w in self.color_frame.winfo_children():
            try: w.config(state=state)
            except: pass

    def _update_matting_sliders(self):
        state = "normal" if self.alpha_matting.get() else "disabled"
        for w in self.matting_frame.winfo_children():
            for ww in w.winfo_children():
                try: ww.config(state=state)
                except: pass

    def _pick_color(self, event=None):
        initial = "#{:02X}{:02X}{:02X}".format(*self.bg_color)
        result = colorchooser.askcolor(color=initial, title="Choose Background Colour")
        if result and result[1]:
            self._set_color(result[1])

    def _set_color(self, hex_color):
        self.bg_color = hex_to_rgb(hex_color)
        self.color_swatch.config(bg=hex_color)
        self.bg_mode.set("color")
        self._update_bg_controls()
        if self.result_image:
            self._refresh_preview()

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        try:
            self.src_image = Image.open(path).convert("RGBA")
            self.src_path = path
            self.result_image = None
            fname = Path(path).name
            self.file_lbl.config(text=fname, fg=C["green"])
            self.run_btn.config(state="normal")
            self.save_png_btn.config(state="disabled")
            self.save_jpg_btn.config(state="disabled")
            self.drop_overlay.place_forget()
            self.view_mode.set("original")
            self._refresh_preview()
            size = self.src_image.size
            self._status(f"Loaded: {fname}  -  {size[0]}x{size[1]} px")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open image:\n{exc}")

    def _save_png(self):
        if not self.result_image:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=Path(self.src_path).stem + "_nobg.png"
        )
        if path:
            self.result_image.save(path)
            self._status(f"Saved PNG -> {path}")

    def _save_jpg(self):
        if not self.result_image:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg")],
            initialfile=Path(self.src_path).stem + "_nobg.jpg"
        )
        if path:
            flat = composite_on_color(self.result_image, self.bg_color)
            flat.save(path, "JPEG", quality=97)
            self._status(f"Saved JPG -> {path}")

    # PROCESSING
    def _start_removal(self):
        if not self.src_image or self.processing:
            return
        self.processing = True
        self.run_btn.config(state="disabled", text="Processing...")
        self.progress.start(12)
        self._status("Loading AI model and removing background...")
        threading.Thread(target=self._remove_bg_thread, daemon=True).start()

    def _remove_bg_thread(self):
        try:
            model_key = self.model_var.get()
            model_name = MODELS[model_key]

            if self.session_name != model_name:
                self._status(f"Loading model: {model_name}  (first run downloads ~170MB)...")
                self.session = new_session(model_name)
                self.session_name = model_name

            self._status("Running AI segmentation...")

            buf = io.BytesIO()
            self.src_image.convert("RGBA").save(buf, format="PNG")
            buf.seek(0)
            data = buf.read()

            kwargs = dict(session=self.session)
            if self.alpha_matting.get():
                kwargs.update(
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=self.fg_thresh.get(),
                    alpha_matting_background_threshold=self.bg_thresh.get(),
                    alpha_matting_erode_size=self.erode_size.get(),
                )

            result_bytes = remove(data, **kwargs)
            self.result_image = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
            self.root.after(0, self._on_done)

        except Exception as exc:
            self.root.after(0, lambda: self._on_error(str(exc)))

    def _on_done(self):
        self.processing = False
        self.progress.stop()
        self.run_btn.config(state="normal", text="Remove Background")
        self.save_png_btn.config(state="normal")
        self.save_jpg_btn.config(state="normal")
        self.view_mode.set("result")
        self._refresh_preview()
        self._status("Done. Use the export buttons to save your image.")

    def _on_error(self, msg):
        self.processing = False
        self.progress.stop()
        self.run_btn.config(state="normal", text="Remove Background")
        self._status(f"Error: {msg}")
        messagebox.showerror("Processing Error", msg)

    # PREVIEW
    def _refresh_preview(self):
        mode = self.view_mode.get()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self.canvas.delete("all")

        if mode == "original" and self.src_image:
            self._draw_single(self.src_image.copy().convert("RGB"), cw, ch)
        elif mode == "result" and self.result_image:
            self._draw_result(self.result_image, cw, ch)
        elif mode == "split" and self.src_image:
            if self.result_image:
                self._draw_split(cw, ch)
            else:
                self._draw_single(self.src_image.copy().convert("RGB"), cw, ch)

    def _draw_single(self, img, cw, ch):
        display = img.copy()
        display.thumbnail((cw - 24, ch - 24), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(display)
        self.canvas._tk_img = tk_img
        x, y = cw // 2, ch // 2
        self.canvas.create_image(x, y, image=tk_img, anchor="center")

    def _draw_result(self, img, cw, ch):
        if self.bg_mode.get() == "color":
            display = composite_on_color(img, self.bg_color)
        else:
            checker = make_checkerboard(img.width, img.height)
            checker = checker.convert("RGBA")
            checker.paste(img, mask=img.split()[3])
            display = checker.convert("RGB")
        display.thumbnail((cw - 24, ch - 24), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(display)
        self.canvas._tk_img = tk_img
        self.canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")

    def _draw_split(self, cw, ch):
        src = self.src_image.copy().convert("RGB")
        res = self.result_image.copy()

        w = min(src.width, res.width)
        h = min(src.height, res.height)
        src = src.crop((0, 0, w, h))
        res = res.crop((0, 0, w, h))

        if self.bg_mode.get() == "color":
            res_rgb = composite_on_color(res, self.bg_color)
        else:
            checker = make_checkerboard(w, h)
            checker = checker.convert("RGBA")
            checker.paste(res, mask=res.split()[3])
            res_rgb = checker.convert("RGB")

        split_img = Image.new("RGB", (w * 2, h))
        split_img.paste(src, (0, 0))
        split_img.paste(res_rgb, (w, 0))

        draw = ImageDraw.Draw(split_img)
        draw.line([(w, 0), (w, h)], fill="#7C6AF7", width=3)

        split_img.thumbnail((cw - 24, ch - 24), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(split_img)
        self.canvas._tk_img = tk_img
        self.canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")

        self.canvas.create_text(40, 18, text="ORIGINAL",
                                font=FONTS["badge"], fill="#FFFFFF", anchor="nw")
        self.canvas.create_text(split_img.width // 2 + 46, 18, text="RESULT",
                                font=FONTS["badge"], fill=C["green"], anchor="nw")

    def _status(self, msg):
        self.root.after(0, lambda: self.status_lbl.config(text=f"  {msg}"))


def main():
    root = tk.Tk()

    # Dark title bar on Windows
    try:
        from ctypes import windll, byref, sizeof, c_int
        HWND = windll.user32.GetParent(root.winfo_id())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        windll.dwmapi.DwmSetWindowAttribute(HWND, DWMWA_USE_IMMERSIVE_DARK_MODE,
                                            byref(c_int(1)), sizeof(c_int))
    except Exception:
        pass

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=C["border"], background=C["accent"],
                    thickness=4, borderwidth=0)

    app = BGEraserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
