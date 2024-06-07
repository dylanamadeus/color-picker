import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Fungsi untuk ekstraksi warna dominan
def extract_dominant_colors(image, k=5):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = colors[sorted_indices]
    return dominant_colors

# CSS untuk mempercantik tampilan
st.markdown(
    """
    <style>
    .color-block {
        display: inline-block;
        width: 100px;
        height: 100px;
        margin-right: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .color-label {
        margin-top: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul aplikasi
st.title('Dominant Color Picker')

# Unggah gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Tampilkan gambar yang diunggah
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Ekstraksi warna dominan
    st.write("Extracting dominant colors...")
    dominant_colors = extract_dominant_colors(image, k=5)

    # Tampilkan warna dominan
    st.write("Dominant colors:")
    cols = st.columns(5)  # Create columns for better layout
    for i, color in enumerate(dominant_colors):
        color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        with cols[i]:
            st.markdown(
                f'<div class="color-block" style="background-color: rgb({color[0]}, {color[1]}, {color[2]});"></div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="color-label">RGB: {color}<br>HEX: {color_hex}</div>', unsafe_allow_html=True)
