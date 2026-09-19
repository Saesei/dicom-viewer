import streamlit as st
import numpy as np
import pydicom

st.set_page_config(layout="wide", page_title="DICOM & NPZ Viewer")
st.title("Interactive CT Volume Viewer")

# File uploader accepts both DICOM files and NPZ archives
uploaded_files = st.file_uploader(
    "Upload DICOM slices (.dcm) OR a compressed dataset (.npz)", 
    accept_multiple_files=True,
    type=["dcm", "npz"]
)

volume = None
center, width = 40.0, 400.0

if uploaded_files:
    # Check if an NPZ file was uploaded
    npz_files = [f for f in uploaded_files if f.name.endswith('.npz')]
    
    if npz_files:
        # Load directly from .npz file
        try:
            data = np.load(npz_files[0])
            volume = data['volume']
            if 'center' in data and 'width' in data:
                center, width = float(data['center']), float(data['width'])
            st.sidebar.success(f"Loaded dataset from: {npz_files[0].name}")
        except Exception as e:
            st.error(f"Error reading .npz file: {e}")

    else:
        # Load from multiple DICOM files
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
            st.sidebar.success(f"Loaded {len(slices)} DICOM slices")

if volume is not None:
    nz, ny, nx = volume.shape

    st.sidebar.header("Windowing Controls")
    
    # Presets
    preset = st.sidebar.selectbox("Preset", ["Default / Custom", "Soft Tissue (40 / 400)", "Bone (400 / 1800)", "Lungs (-600 / 1500)"])
    
    if preset == "Soft Tissue (40 / 400)":
        center, width = 40.0, 400.0
    elif preset == "Bone (400 / 1800)":
        center, width = 400.0, 1800.0
    elif preset == "Lungs (-600 / 1500)":
        center, width = -600.0, 1500.0

    center = st.sidebar.slider("Window Level (HU)", -1000, 1000, int(center))
    width = st.sidebar.slider("Window Width (HU)", 1, 3000, int(width))

    min_v = center - (width / 2.0)
    max_v = center + (width / 2.0)

    # 3-Panel View Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        z_idx = st.slider("Axial (Z)", 0, nz - 1, nz // 2)
        img_ax = np.clip(volume[z_idx, :, :], min_v, max_v)
        st.image(img_ax, caption=f"Axial (Slice {z_idx + 1}/{nz})", clamp=True, use_container_width=True)

    with col2:
        y_idx = st.slider("Coronal (Y)", 0, ny - 1, ny // 2)
        img_cor = np.clip(volume[:, y_idx, :], min_v, max_v)
        st.image(np.flipud(img_cor), caption=f"Coronal (Slice {y_idx + 1}/{ny})", clamp=True, use_container_width=True)

    with col3:
        x_idx = st.slider("Sagittal (X)", 0, nx - 1, nx // 2)
        img_sag = np.clip(volume[:, :, x_idx], min_v, max_v)
        st.image(np.flipud(img_sag), caption=f"Sagittal (Slice {x_idx + 1}/{nx})", clamp=True, use_container_width=True)
