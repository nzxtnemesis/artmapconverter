# === Imports ===
import streamlit as st
import tempfile
import shutil
import os
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np

# === Palette Loader ===
def load_fupery_palette():
    # Loads and parses the CSV palette
    df = pd.read_csv("full_artmap_dye_palette.csv")
    palette = {}
    for _, row in df.iterrows():
        base = row['Item'].lower().replace(" ", "_")
        for i in range(1, 5):
            label = f"{base}{i}"
            hex_code = row[f'Color{i}']
            if pd.notna(hex_code):
                hex_code = hex_code.strip().lstrip('#')
                if len(hex_code) != 6:
                    continue  # skip malformed color
                r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
                palette[label] = (r, g, b)
    return palette

# === Canvas Preview Builder ===
def build_preview_image(tiles, palette, grid_w, grid_h, tile_size=32):
    # Combines all individual tile images into a single large preview image
    preview_img = Image.new("RGB", (grid_w * tile_size, grid_h * tile_size))
    for gx, gy, _, tile_img in tiles:
        preview_img.paste(tile_img, ((gx - 1) * tile_size, (gy - 1) * tile_size))
    return preview_img

# === Dithering Utilities ===
def apply_dithering(img, palette, method):
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape

    def get_closest(p):
        return np.array(min(palette.values(), key=lambda c: sum((p[i] - c[i])**2 for i in range(3))))

    def distribute_error(x, y, err, weights):
        for dx, dy, weight in weights:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                arr[ny, nx] += err * weight

    weights = {
        "Floyd–Steinberg": [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)],
        "Burkes": [(1, 0, 8/32), (2, 0, 4/32), (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 8/32), (1, 1, 4/32), (2, 1, 2/32)],
        "Sierra-Lite": [(1, 0, 2/4), (-1, 1, 1/4), (0, 1, 1/4)]
    }[method]

    for y in range(h):
        for x in range(w):
            old = arr[y, x].copy()
            new = get_closest(old)
            arr[y, x] = new
            distribute_error(x, y, old - new, weights)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def closest_fupery_dye(rgb, palette):
    min_dist = float('inf')
    closest_label = ""
    for label, dye_rgb in palette.items():
        dist = sum((p - q) ** 2 for p, q in zip(rgb, dye_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_label = label
    return closest_label

def find_exact_label(rgb, palette, rgb_to_label):
    return rgb_to_label.get(rgb, closest_fupery_dye(rgb, palette))

def process_image_with_palette(img, grid_w, grid_h, palette, tile_size=32, dither_method=None):
    rgb_to_label = {tuple(v): k for k, v in palette.items()}
    img = img.resize((grid_w * tile_size, grid_h * tile_size), Image.LANCZOS)
    if dither_method:
        img = apply_dithering(img, palette, dither_method)

    tiles = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            tile = img.crop((gx * tile_size, gy * tile_size, (gx + 1) * tile_size, (gy + 1) * tile_size))
            dye_grid = []
            for y in range(tile_size):
                row = []
                for x in range(tile_size):
                    pixel = tile.getpixel((x, y))
                    dye_name = find_exact_label(pixel, palette, rgb_to_label)
                    row.append(dye_name)
                dye_grid.append(row)
            tiles.append((gx + 1, gy + 1, pd.DataFrame(dye_grid), tile))
    return tiles

# === Dye Usage Summary Generator ===
def generate_dye_step_summary(tiles):
    # Summarizes dye usage per tile and overall
    summary = {}
    tile_dye_map = {}
    for gx, gy, df, _ in tiles:
        dye_counts = {}
        for row in df.values:
            for dye in row:
                dye_base = dye.rstrip("1234")
                dye_counts[dye_base] = dye_counts.get(dye_base, 0) + 1
                summary[dye_base] = summary.get(dye_base, 0) + 1
        tile_dye_map[f"map_{gx}_{gy}"] = dye_counts
    return summary, tile_dye_map

# === Streamlit UI ===
st.set_page_config(page_title="🎨 ArtMap Image Converter", layout="wide")
st.title("🎨 ArtMap Image Converter")

# UI controls
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
grid_w = st.number_input("Canvas Width (tiles)", min_value=1, max_value=25, value=4)
grid_h = st.number_input("Canvas Height (tiles)", min_value=1, max_value=25, value=4)

col1, col2, col3, col4 = st.columns(4)
with col1:
    brightness = st.slider("🔆 Brightness", 0.1, 2.0, 1.0, 0.1)
with col2:
    contrast = st.slider("🌓 Contrast", 0.1, 2.0, 1.0, 0.1)
with col3:
    saturation = st.slider("🎨 Saturation", 0.1, 2.0, 1.0, 0.1)
with col4:
    sharpness = st.slider("✴️ Sharpness", 0.1, 2.0, 1.0, 0.1)

filter_option = st.selectbox("🎨 Filter Effect", ["None", "Grayscale", "Sepia"])
zoom_mode = st.radio("🔍 Preview Zoom Mode", ["100% (Pixel-Perfect)", "400% (In-Game Approximate)"], horizontal=True)
dither_method = st.selectbox("🌀 Dithering Method", ["None", "Floyd–Steinberg", "Burkes", "Sierra-Lite"])
process_button = st.button("🧪 Generate Canvas Output")

if uploaded and process_button:
    # Load and adjust image
    image = Image.open(uploaded).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Apply optional effects
    if filter_option == "Grayscale":
        image = image.convert("L").convert("RGB")
    elif filter_option == "Sepia":
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        tr = [0.393, 0.769, 0.189]
        tg = [0.349, 0.686, 0.168]
        tb = [0.272, 0.534, 0.131]
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        sepia = np.stack([
            r * tr[0] + g * tr[1] + b * tr[2],
            r * tg[0] + g * tg[1] + b * tg[2],
            r * tb[0] + g * tb[1] + b * tb[2]
        ], axis=-1).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(sepia)

    # Boost contrast only when dithering is active
    dither = None if dither_method == "None" else dither_method
    if dither:
        image = ImageEnhance.Contrast(image).enhance(1.5)

    palette = load_fupery_palette()

    # Process the image with the configured settings
    tiles = process_image_with_palette(image, grid_w, grid_h, palette, dither_method=dither)

    # Show final canvas output
    preview = build_preview_image(tiles, palette, grid_w, grid_h)
    zoom_factor = 1 if zoom_mode == "100% (Pixel-Perfect)" else 4
    resized = preview.resize((preview.width * zoom_factor, preview.height * zoom_factor), Image.NEAREST)
    st.image(resized, caption="🖼️ Canvas Preview")

    # Display dye usage summary
    with st.expander("🎨 View Dyes Used Per Tile", expanded=False):
        summary, per_tile = generate_dye_step_summary(tiles)
        cols = st.columns(3)
        sorted_tiles = sorted(per_tile.items())
        for i, (tile, dyes) in enumerate(sorted_tiles):
            dye_list = sorted(dyes.keys())
            with cols[i % 3]:
                for gx, gy, _, tile_img in tiles:
                    if f"map_{gx}_{gy}" == tile:
                        st.image(tile_img.resize((96, 96)), use_container_width=False)
                        st.markdown(f"**{tile}**: " + ", ".join(f"`{d}`" for d in dye_list))
                        break

    # Export CSV + PNG tile pack
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for gx, gy, df, tile_img in tiles:
            df.to_csv(os.path.join(tmpdir, f"map_{gx}_{gy}.csv"), index=False)
            tile_img.save(os.path.join(images_dir, f"map_{gx}_{gy}.png"))

        zip_base = os.path.splitext(uploaded.name)[0]
        zip_out = shutil.make_archive(zip_base, 'zip', tmpdir)
        with open(zip_out, "rb") as f:
            st.download_button("📦 Download Map CSVs + Images", f, file_name=f"{zip_base}_{grid_w}x{grid_h}.zip")
