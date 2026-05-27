from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans


APP_TITLE = "Color Palette AI"
PALETTE_SIZE = 5
MAX_SAMPLE_PIXELS = 60_000


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = [int(value) for value in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def get_image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_pixels(image: Image.Image) -> np.ndarray:
    rgba_image = image.convert("RGBA")
    pixels = np.asarray(rgba_image).reshape(-1, 4)

    opaque_pixels = pixels[pixels[:, 3] > 20][:, :3]
    if len(opaque_pixels) == 0:
        opaque_pixels = pixels[:, :3]

    if len(opaque_pixels) > MAX_SAMPLE_PIXELS:
        rng = np.random.default_rng(42)
        sample_index = rng.choice(len(opaque_pixels), MAX_SAMPLE_PIXELS, replace=False)
        opaque_pixels = opaque_pixels[sample_index]

    return opaque_pixels.astype(np.float32)


@st.cache_data(show_spinner=False)
def extract_palette(image_bytes: bytes, clusters: int = PALETTE_SIZE) -> pd.DataFrame:
    image = Image.open(BytesIO(image_bytes))
    pixels = prepare_pixels(image)
    unique_colors = np.unique(pixels.astype(np.uint8), axis=0)
    cluster_count = min(clusters, len(unique_colors))

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    counts = np.bincount(labels, minlength=cluster_count)
    order = np.argsort(counts)[::-1]
    total = counts.sum()

    rows = []
    for rank, cluster_id in enumerate(order, start=1):
        rgb = tuple(int(value) for value in centers[cluster_id])
        rows.append(
            {
                "rank": rank,
                "hex": rgb_to_hex(rgb),
                "rgb": f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                "red": rgb[0],
                "green": rgb[1],
                "blue": rgb[2],
                "dominance": round((counts[cluster_id] / total) * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def color_card(hex_value: str, rgb_value: str, dominance: float) -> str:
    red = int(hex_value[1:3], 16)
    green = int(hex_value[3:5], 16)
    blue = int(hex_value[5:7], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    text_color = "#151515" if luminance > 0.62 else "#F7F8F6"
    subtext_color = (
        "rgba(21,21,21,0.62)"
        if text_color == "#151515"
        else "rgba(247,248,246,0.72)"
    )

    return f"""
    <div style="
        min-height:184px;
        height:184px;
        width:100%;
        display:flex;
        flex-direction:column;
        justify-content:flex-end;
        gap:6px;
        padding:22px 24px;
        border-radius:28px;
        background:{hex_value};
        box-sizing:border-box;">
        <strong style="
            color:{text_color};
            font-size:1.35rem;
            font-weight:800;
            line-height:1.05;
            letter-spacing:0;">
            {hex_value.lower()}
        </strong>
        <span style="
            color:{subtext_color};
            font-size:0.96rem;
            font-weight:750;
            line-height:1.2;
            white-space:nowrap;">
            {dominance:.2f}%
        </span>
    </div>
    """


def render_empty_state() -> None:
    st.info(
        "Unggah gambar JPG, PNG, atau WEBP untuk melihat lima warna paling dominan "
        "berdasarkan K-Means Clustering."
    )

st.title("Color Palette AI")
st.header("Deskripsi Singkat")
st.write(
    """
    Aplikasi ini mengubah gambar menjadi palet lima warna dominan menggunakan algoritma K-Means Clustering. Cocok untuk desain UI, branding, moodboard, dan eksplorasi warna yang cepat.
    """
)

st.header("Penjelasan K-Means Clustering")
st.write(
    """
    **K-Means Clustering** adalah algoritma unsupervised learning yang digunakan untuk mengelompokkan data ke dalam beberapa cluster berdasarkan kemiripan fitur. Pada aplikasi ini, setiap piksel gambar dianggap sebagai data dengan tiga fitur (R, G, B). K-Means akan mengelompokkan piksel-piksel tersebut ke dalam beberapa cluster (misal 5), lalu pusat cluster (centroid) dianggap sebagai warna dominan.

    **Penerapan pada program:**
    1. Gambar diubah menjadi array piksel RGB.
    2. Piksel diambil sampelnya dan dikelompokkan menggunakan K-Means.
    3. Setiap cluster menghasilkan satu warna dominan.
    4. Warna-warna dominan diurutkan berdasarkan jumlah piksel yang termasuk dalam cluster tersebut.
    """
)

st.header("Unggah Gambar")
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "webp"],
    help="Gambar dengan warna jelas akan menghasilkan palet yang lebih akurat."
)

if uploaded_file is None:
    st.info(
        "Unggah gambar JPG, PNG, atau WEBP untuk melihat lima warna paling dominan berdasarkan K-Means Clustering."
    )
    st.stop()


with st.sidebar:
    st.header("Daftar Isi")
    st.markdown("""
    - [Deskripsi Singkat](#deskripsi-singkat)
    - [Penjelasan K-Means Clustering](#penjelasan-k-means-clustering)
    - [Unggah Gambar](#unggah-gambar)
    - [Gambar](#gambar)
    - [Color Palette dari Gambar](#color-palette-dari-gambar)
    - [Data Palet](#data-palet)
    """)
    st.markdown("""
    <hr style='margin-top:2em;margin-bottom:0.5em;'>
    <div style='text-align:center; color:gray; font-size:0.95em;'>
        Dibuat oleh:
                <br>
                <b>Muhammad Fathin Raihan - 140810240026</b> &copy; 2026
    </div>
    """, unsafe_allow_html=True)


if uploaded_file is None:
    render_empty_state()
    st.stop()

image = Image.open(uploaded_file)
image_bytes = get_image_bytes(image)
palette = extract_palette(image_bytes)

st.header("Gambar")
st.image(image, caption=uploaded_file.name, use_container_width=True)
st.write(f"Ukuran: {image.width} x {image.height}px")
st.write(f"Cluster: {len(palette)} warna")
st.write("Algoritma: K-Means")

st.header("Color Palette dari Gambar")

card_columns = st.columns(len(palette))
for column, row in zip(card_columns, palette.itertuples()):
    with column:
        st.markdown(color_card(row.hex, row.rgb, float(row.dominance)), unsafe_allow_html=True)

st.header("Data Palet")
st.dataframe(
    palette[["rank", "hex", "rgb", "dominance"]],
    use_container_width=True,
    hide_index=True,
)

csv_data = palette.to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh palet sebagai CSV",
    data=csv_data,
    file_name="dominant_color_palette.csv",
    mime="text/csv",
    use_container_width=True,
)
st.caption("Warna pada color picker dapat disesuaikan setelah palet terbentuk.")
