import streamlit as st
import numpy as np
import pydicom
from scipy.ndimage import zoom

st.set_page_config(layout="wide", page_title="CT DICOM Viewer")

PRESETS = {
    'Custom / Default': None,
    'Soft Tissue (Abdomen/Brain)': (40.0, 400.0),
    'Bone': (400.0, 1800.0),
    'Lungs': (-600.0, 1500.0)
}

def apply_windowing(image, center, width):
    min_val = center - (width / 2.0)
    max_val = center + (width / 2.0)
    clipped = np.clip(image, min_val, max_val)
    if max_val != min_val:
        return (clipped - min_val) / (max_val - min_val)
    return clipped

st.title("Interactive CT Volume Viewer")

uploaded_files = st.file_uploader(
    "Upload DICOM files (.dcm) OR a compressed cache (.npz)", 
    accept_multiple_files=True,
    type=["dcm", "npz"]
)

volume = None
default_center, default_width = 40.0, 400.0
spacing_z, spacing_y, spacing_x = 1.0, 1.0, 1.0

if uploaded_files:
    npz_files = [f for f in uploaded_files if f.name.endswith('.npz')]
    
    if npz_files:
        try:
            data = np.load(npz_files[0])
            volume = data['volume']
            if 'center' in data and 'width' in data:
                default_center, default_width = float(data['center']), float(data['width'])
            if 'spacing' in data:
                spacing_z, spacing_y, spacing_x = [float(s) for s in data['spacing']]
            st.sidebar.success(f"Loaded cache: {npz_files[0].name}")
        except Exception as e:
            st.error(f"Error loading .npz file: {e}")

    else:
        slices = []
        for file in uploaded_files:
            try:
                ds = pydicom.dcmread(file, force=True)
                if not hasattr(ds, 'file_meta') or 'TransferSyntaxUID' not in ds.file_meta:
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                continue

        if slices:
            try:
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            except (AttributeError, KeyError):
                try:
                    slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
                except (ValueError, TypeError):
                    slices.sort(key=lambda x: x.filename)

            try:
                spacing_y, spacing_x = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])
            except (AttributeError, KeyError):
                spacing_y, spacing_x = 1.0, 1.0

            try:
                if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
                    spacing_z = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
                else:
                    spacing_z = float(getattr(slices[0], 'SliceThickness', 1.0))
            except Exception:
                spacing_z = 1.0

            hu_slices = []
            for s in slices:
                slope = float(getattr(s, 'RescaleSlope', 1))
                intercept = float(getattr(s, 'RescaleIntercept', 0))
                hu_slice = (s.pixel_array.astype(np.float32) * slope) + intercept
                hu_slices.append(hu_slice)

            volume = np.stack(hu_slices)

            if hasattr(slices[0], 'WindowCenter') and hasattr(slices[0], 'WindowWidth'):
                try:
                    wc, ww = slices[0].WindowCenter, slices[0].WindowWidth
                    default_center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
                    default_width = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
                except Exception:
                    pass

            st.sidebar.success(f"Loaded {len(slices)} DICOM slices")

if volume is not None:
    nz, ny, nx = volume.shape

    # Calculate exact vertical scaling multipliers relative to horizontal pixel width
    aspect_coronal = spacing_z / spacing_x
    aspect_sagittal = spacing_z / spacing_y

    st.sidebar.header("Controls & Windowing")
    
    selected_preset = st.sidebar.selectbox("Presets", list(PRESETS.keys()))
    if PRESETS[selected_preset] is not None:
        p_center, p_width = PRESETS[selected_preset]
    else:
        p_center, p_width = default_center, default_width

    center = st.sidebar.slider("Level (HU)", -1000, 1000, int(p_center))
    width = st.sidebar.slider("Width (HU)", 1, 3000, int(p_width))

    col1, col2, col3 = st.columns(3)

    # 1. Axial View (Z) - Standard pixel grid
    with col1:
        z_idx = st.slider("Z (Axial)", 0, nz - 1, nz // 2)
        slice_ax = apply_windowing(volume[z_idx, :, :], center, width)
        st.image(slice_ax, caption=f"Axial (Z: {z_idx + 1}/{nz})", use_container_width=True)

    # 2. Coronal View (Y) - Rescaled vertically by aspect_coronal
    with col2:
        y_idx = st.slider("Y (Coronal)", 0, ny - 1, ny // 2)
        slice_cor = apply_windowing(volume[:, y_idx, :], center, width)
        slice_cor = np.flipud(slice_cor)
        
        # Resample array height to match real physical aspect ratio
        if aspect_coronal != 1.0:
            slice_cor = zoom(slice_cor, (aspect_coronal, 1.0), order=1)
            
        st.image(slice_cor, caption=f"Coronal (Y: {y_idx + 1}/{ny})", use_container_width=True)

    # 3. Sagittal View (X) - Rescaled vertically by aspect_sagittal
    with col3:
        x_idx = st.slider("X (Sagittal)", 0, nx - 1, nx // 2)
        slice_sag = apply_windowing(volume[:, :, x_idx], center, width)
        slice_sag = np.flipud(slice_sag)
        
        # Resample array height to match real physical aspect ratio
        if aspect_sagittal != 1.0:
            slice_sag = zoom(slice_sag, (aspect_sagittal, 1.0), order=1)
            
        st.image(slice_sag, caption=f"Sagittal (X: {x_idx + 1}/{nx})", use_container_width=True)
