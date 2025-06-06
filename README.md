# 🎨 ArtMap Image Converter

A Streamlit web app that converts any image into Minecraft-compatible map art tiles using the full Fupery dye palette. Includes dithering support, color enhancements, tile breakdowns, and downloadable map kits.

## ✨ Features

- Upload any image (JPG/PNG)
- Choose canvas grid size (e.g., 3x4 maps)
- Adjust brightness, contrast, saturation, and sharpness
- Apply popular dithering algorithms (Floyd–Steinberg, Burkes, Sierra-Lite)
- Auto-slice canvas into per-map CSV and image tiles
- Preview your output at full or zoomed resolution
- View dyes used per map tile
- Download CSV + PNG image kit in one ZIP

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Files

- `app.py`: Main Streamlit application (formerly `Artmap_1.1.py`)
- `full_artmap_dye_palette.csv`: Color palette file with Fupery's dye shades
- `requirements.txt`: Dependency list

---

Built with ❤️ to make Minecraft map art faster, easier, and more vibrant.
