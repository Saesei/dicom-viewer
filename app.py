import streamlit as st
import numpy as np
import pydicom

st.set_page_config(layout="wide", page_title="DICOM Viewer")
st.title("Interactive CT DICOM Viewer")

# File uploader allows selecting multiple DICOM files at once
uploaded_files = st.file_uploader(
    "Upload DICOM scan files (.dcm)", 
    accept_multiple_files=True
)

if uploaded_files:
    slices = []
    for file in uploaded_files:
        try:
            ds = pydicom.dcmread(file, force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if slices:
        # Sort spatially by Z position or Instance Number
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except Exception:
            slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))

        # Convert to Hounsfield Units (HU)
        hu_slices = []
        for s in slices:
            slope = float(getattr(s, 'RescaleSlope', 1))
            intercept = float(getattr(s, 'RescaleIntercept', 0))
            hu_slice = (s.pixel_array.astype(np.float32) * slope) + intercept
            hu_slices.append(hu_slice)

        volume = np.stack(hu_slices)
        nz, ny, nx = volume.shape

        st.sidebar.header("Controls")
        
        # Windowing Presets
        preset = st.sidebar.selectbox("Windowing Preset", ["Custom", "Soft Tissue (40 / 400)", "Bone (400 / 1800)", "Lungs (-600 / 1500)"])
        
        if preset == "Soft Tissue (40 / 400)":
            default_center, default_width = 40, 400
        elif preset == "Bone (400 / 1800)":
            default_center, default_width = 400, 1800
        elif preset == "Lungs (-600 / 1500)":
            default_center, default_width = -600, 1500
        else:
            default_center, default_width = 40, 400

        center = st.sidebar.slider("Window Level (HU)", -1000, 1000, default_center)
        width = st.sidebar.slider("Window Width (HU)", 1, 3000, default_width)

        col1, col2, col3 = st.columns(3)

        with col1:
            z_idx = st.slider("Axial (Z)", 0, nz - 1, nz // 2)
            min_v = center - (width / 2.0)
            max_v = center + (width / 2.0)
            img_ax = np.clip(volume[z_idx, :, :], min_v, max_v)
            st.image(img_ax, caption=f"Axial View (Slice {z_idx + 1}/{nz})", clamp=True, use_container_width=True)

        with col2:
            y_idx = st.slider("Coronal (Y)", 0, ny - 1, ny // 2)
            img_cor = np.clip(volume[:, y_idx, :], min_v, max_v)
            st.image(np.flipud(img_cor), caption=f"Coronal View (Slice {y_idx + 1}/{ny})", clamp=True, use_container_width=True)

        with col3:
            x_idx = st.slider("Sagittal (X)", 0, nx - 1, nx // 2)
            img_sag = np.clip(volume[:, :, x_idx], min_v, max_v)
            st.image(np.flipud(img_sag), caption=f"Sagittal View (Slice {x_idx + 1}/{nx})", clamp=True, use_container_width=True)
    else:
        st.error("No valid DICOM image files detected.")
