



# app.py (fixed: unique stacked/download keys + combined molecular-band multiselect)
import streamlit as st
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import pandas as pd
import tempfile, os, io, time, re
from typing import Tuple
import plotly.graph_objects as go

st.set_page_config(page_title="AstroFlow · FITSFlow", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper: stable key generator
# ---------------------------
def make_key(*parts):
    raw = "_".join(str(p) for p in parts if p is not None)
    key = re.sub(r'\W+', '_', raw).strip('_')
    return key[:200]

# ---------------------------
# Helper / Processing Utils
# ---------------------------
WL_COLS = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WLEN', 'LAMBDA_MICRON', 'LAMBDA_UM', 'WAVELENGTH_MICRON']
FLUX_COLS = ['FLUX', 'FLUX_DENSITY', 'SPECTRUM', 'INTENSITY', 'FLUX_1', 'FLUX_0']

def safe_names(arr):
    try:
        return list(arr.names)
    except Exception:
        return []

def find_wl_flux_from_table(table):
    names = safe_names(table)
    wl_col = next((c for c in WL_COLS if c in names), None)
    fl_col = next((c for c in FLUX_COLS if c in names), None)
    return wl_col, fl_col

def try_extract_spectrum(hdu):
    data = hdu.data
    if data is None:
        return None, None
    # Table-like
    if hasattr(data, 'names'):
        wl_col, fl_col = find_wl_flux_from_table(data)
        if wl_col and fl_col:
            wl = np.array(data[wl_col]).astype(float).flatten()
            fl = np.array(data[fl_col]).astype(float).flatten()
            mask = np.isfinite(wl) & np.isfinite(fl)
            return wl[mask], fl[mask]
        # fallback: first two numeric columns
        names = safe_names(data)
        nums = [n for n in names if np.issubdtype(data[n].dtype, np.number)]
        if len(nums) >= 2:
            wl = np.array(data[nums[0]]).astype(float).flatten()
            fl = np.array(data[nums[1]]).astype(float).flatten()
            mask = np.isfinite(wl) & np.isfinite(fl)
            return wl[mask], fl[mask]
    # Image-like
    try:
        arr = np.array(data)
        if arr.ndim == 1:
            wl = np.arange(arr.size)
            fl = arr.astype(float)
            mask = np.isfinite(fl)
            return wl[mask], fl[mask]
        elif arr.ndim == 2:
            fl = np.nanmean(arr, axis=0)
            wl = np.arange(fl.size)
            mask = np.isfinite(fl)
            return wl[mask], fl[mask]
    except Exception:
        pass
    return None, None

def interp_to_reference(wl, fl, ref_wl):
    try:
        return np.interp(ref_wl, wl, fl, left=np.nan, right=np.nan)
    except Exception:
        return np.full_like(ref_wl, np.nan)

def smooth_flux(flux, window, polyorder):
    if len(flux) >= window and window % 2 == 1:
        return savgol_filter(flux, window, polyorder)
    return flux

def calc_snr_on_band(ref_wl, ref_flux, band_range: Tuple[float,float]):
    start, end = band_range
    mask = (ref_wl >= start) & (ref_wl <= end)
    if not np.any(mask):
        return 0.0
    signal = np.abs(1 - np.nanmean(ref_flux[mask]))
    left_mask = (ref_wl >= (start - 0.3)) & (ref_wl <= (start - 0.1))
    right_mask = (ref_wl >= (end + 0.1)) & (ref_wl <= (end + 0.3))
    noise_vals = []
    if np.any(left_mask):
        noise_vals.append(np.nanstd(ref_flux[left_mask]))
    if np.any(right_mask):
        noise_vals.append(np.nanstd(ref_flux[right_mask]))
    noise = np.nanmean(noise_vals) if noise_vals else np.nanstd(ref_flux)
    if noise == 0 or np.isnan(noise):
        return 0.0
    return float(signal / noise)

# expand molecular band list here if you like
DEFAULT_BANDS = {
    "H2O": (1.35, 1.45),
    "CH4": (1.60, 1.72),
    "CO2": (2.65, 2.75),
    # add more bands as needed
}

# ---------------------------
# Sidebar controls (UI)
# ---------------------------
st.sidebar.header("AstroFlow Controls")
st.sidebar.markdown("Upload FITS files and toggle analysis options.")

smoothing_enabled = st.sidebar.checkbox("Enable smoothing", value=True)
smoothing_window = st.sidebar.slider("Smoothing window (odd)", 5, 501, 51, step=2)
polyorder = st.sidebar.slider("SavGol polyorder", 1, 5, 3)

show_bands = st.sidebar.checkbox("Show molecular bands (overlay)", value=True)
# Replace multiple checkboxes with a single multiselect for clarity & to avoid many widgets
selected_bands = st.sidebar.multiselect(
    "Select molecular bands to display",
    options=list(DEFAULT_BANDS.keys()),
    default=list(DEFAULT_BANDS.keys())
)

show_snr = st.sidebar.checkbox("Show SNR", value=False)
show_errorbars = st.sidebar.checkbox("Show error bars (if available)", value=False)
raw_only = st.sidebar.checkbox("Show raw data only (no smoothing/stacking overlays)", value=False)

stack_enabled = st.sidebar.checkbox("Enable stacking (multi-file)", value=True)
stack_method = st.sidebar.selectbox("Stack method", ["mean", "median"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Display / export options")
enable_downloads = st.sidebar.checkbox("Enable downloads", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Prototype · AstroFlow / FutureMind")

# ---------------------------
# Main UI area
# ---------------------------
st.title("🔭 AstroFlow · FITSFlow Processor")
st.markdown("Upload FITS files (JWST/HST/TESS/generic). Tabs: Raw | Smoothed | Molecule Detection | Stacked | Table")

uploaded = st.file_uploader("Upload one or more FITS files", type=["fits"], accept_multiple_files=True)

if not uploaded:
    st.info("Upload FITS spectral files to start. Example FITS: K2-18b, GJ-1214b.")
    st.stop()

# Save uploaded to temp dir
tmpdir = tempfile.mkdtemp()
file_paths = []
for up in uploaded:
    dst = os.path.join(tmpdir, up.name)
    with open(dst, "wb") as f:
        f.write(up.read())
    file_paths.append(dst)

# Process files with progress
results = []
progress = st.progress(0)
nfiles = len(file_paths)
i = 0

for path in file_paths:
    i += 1
    progress.progress(int((i-1)/nfiles*100))
    fname = os.path.basename(path)
    try:
        with fits.open(path, memmap=False) as hdul:
            primary_hdr = dict(hdul[0].header)
            found_any = False
            for idx, hdu in enumerate(hdul):
                wl, fl = try_extract_spectrum(hdu)
                if wl is None:
                    continue
                found_any = True
                err = None
                results.append({
                    "file": fname,
                    "path": path,
                    "hdu_index": idx,
                    "header": dict(hdu.header),
                    "wl": np.array(wl, dtype=float),
                    "fl": np.array(fl, dtype=float),
                    "err": err
                })
            if not found_any:
                st.warning(f"No 1D spectrum auto-extracted from {fname}. Showing HDU summaries.")
    except Exception as e:
        st.error(f"Failed to open {fname}: {e}")

progress.progress(100)
time.sleep(0.2)

if len(results) == 0:
    st.error("No spectra could be extracted from uploaded files. You may upload pre-processed wavelength+flux CSVs.")
    st.stop()

tabs = st.tabs([
    "Raw Spectrum",
    "Smoothed",
    "Molecule Detection",
    "Stacked",
    "Data Table",
    "Downloads",
    "Images",
    "Reports"
])


def plot_spectrum_interactive(wl, fl, fl_smooth=None, err=None, title="Spectrum", bands=None, show_bands_flag=True, show_error=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=fl, mode='lines', name='raw', line=dict(color='rgba(0,150,200,0.7)')))
    if fl_smooth is not None:
        fig.add_trace(go.Scatter(x=wl, y=fl_smooth, mode='lines', name='smoothed', line=dict(color='black', width=2)))
    if show_error and err is not None:
        fig.add_trace(go.Scatter(x=wl, y=fl+err, mode='lines', name='err+', line=dict(width=0), showlegend=False, opacity=0.2))
        fig.add_trace(go.Scatter(x=wl, y=fl-err, mode='lines', name='err-', line=dict(width=0), showlegend=False, opacity=0.2))
    if show_bands_flag and bands:
        for mol,(a,b) in bands.items():
            fig.add_vrect(x0=a, x1=b, fillcolor="LightSkyBlue", opacity=0.25, layer="below", line_width=0, annotation_text=mol, annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Wavelength", yaxis_title="Flux", template="plotly_white", height=400)
    return fig

# Raw tab
with tabs[0]:
    st.header("Raw Spectrum")
    for res in results:
        label = f"{res['file']} (HDU {res['hdu_index']})"
        # DO NOT pass a key to expander here to avoid Streamlit TypeError in some runtimes
        with st.expander(label, expanded=False):
            st.subheader("Header (partial)")
            hdr = res['header']
            keys_to_show = {k: hdr[k] for k in list(hdr.keys())[:20]}
            st.json(keys_to_show)
            wl = res['wl']; fl = res['fl']; err = res['err']
            fig = plot_spectrum_interactive(wl, fl, fl_smooth=None, err=err, title=label, bands=None, show_bands_flag=False)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'raw')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            st.write(f"Data points: {len(wl)} | Wavelength range: {wl.min():.3g} – {wl.max():.3g}")
            if enable_downloads:
                df = pd.DataFrame({"wavelength": wl, "flux": fl})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'raw_csv')
                st.download_button(f"Download CSV (raw) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_raw.csv", mime='text/csv', key=dl_key)

# Smoothed tab
with tabs[1]:
    st.header("Smoothed Spectra")
    for res in results:
        label = f"{res['file']} (HDU {res['hdu_index']})"
        with st.expander(label, expanded=False):
            wl = res['wl']; fl = res['fl']; err = res['err']
            if raw_only:
                st.info("Raw-only mode enabled. Toggle off to see smoothing.")
                fl_smooth = None
            else:
                fl_proc = fl.copy()
                fl_smooth = smooth_flux(fl_proc, smoothing_window, polyorder) if smoothing_enabled else None
            fig = plot_spectrum_interactive(wl, fl, fl_smooth=fl_smooth, err=err, title=label, bands=None, show_bands_flag=False, show_error=show_errorbars)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'smooth')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            if enable_downloads:
                df = pd.DataFrame({"wavelength": wl, "flux": fl, "flux_smoothed": fl_smooth if fl_smooth is not None else fl})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'smooth_csv')
                st.download_button(f"Download CSV (smoothed) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_smoothed.csv", mime='text/csv', key=dl_key)

# Molecule Detection tab
with tabs[2]:
    st.header("Molecule Detection (band overlays)")
    # Build active bands from the single multiselect and master show_bands toggle
    active_bands = {mol: DEFAULT_BANDS[mol] for mol in selected_bands} if show_bands else {}

    for res in results:
        label = f"{res['file']} (HDU {res['hdu_index']})"
        with st.expander(label, expanded=False):
            wl = res['wl']; fl = res['fl']
            if raw_only:
                fl_proc = fl
            else:
                fl_proc = smooth_flux(fl, smoothing_window, polyorder) if smoothing_enabled else fl
            fig = plot_spectrum_interactive(wl, fl, fl_smooth=fl_proc, err=res['err'], title=label, bands=active_bands, show_bands_flag=show_bands and not raw_only, show_error=show_errorbars)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'mol')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            if show_snr and active_bands:
                snr_table = {mol: calc_snr_on_band(wl, fl_proc, rng) for mol,rng in active_bands.items()}
                st.subheader("SNR (approx)")
                st.json({k: float(np.round(v,3)) for k,v in snr_table.items()})
            if enable_downloads:
                df = pd.DataFrame({"wavelength": wl, "flux": fl, "flux_processed": fl_proc})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'mol_csv')
                st.download_button(f"Download CSV (processed) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_processed.csv", mime='text/csv', key=dl_key)

# Stacked tab
with tabs[3]:
    st.header("Stacked Spectrum")
    if len(results) < 2 or not stack_enabled:
        st.info("Upload multiple spectra and enable stacking to see combined results.")
    else:
        min_wl = min(np.nanmin(r['wl']) for r in results)
        max_wl = max(np.nanmax(r['wl']) for r in results)
        ref_wl = np.linspace(min_wl, max_wl, 2000)
        interp_fluxes = [interp_to_reference(r['wl'], r['fl'], ref_wl) for r in results]
        arr = np.array(interp_fluxes)
        stacked = np.nanmedian(arr, axis=0) if stack_method == "median" else np.nanmean(arr, axis=0)
        if smoothing_enabled and not raw_only and len(stacked) >= smoothing_window:
            stacked_smooth = smooth_flux(stacked, smoothing_window, polyorder)
        else:
            stacked_smooth = stacked
        if True and not raw_only:  # normalization on for visual clarity
            if np.nanmax(stacked_smooth) != np.nanmin(stacked_smooth):
                stacked_norm = (stacked - np.nanmin(stacked)) / (np.nanmax(stacked) - np.nanmin(stacked))
                stacked_smooth = (stacked_smooth - np.nanmin(stacked_smooth)) / (np.nanmax(stacked_smooth) - np.nanmin(stacked_smooth)) if np.nanmax(stacked_smooth) != np.nanmin(stacked_smooth) else stacked_smooth
            else:
                stacked_norm = stacked
        else:
            stacked_norm = stacked

        bands_for_plot = {}
        if "H2O" in selected_bands: bands_for_plot["H2O"] = DEFAULT_BANDS["H2O"]
        if "CH4" in selected_bands: bands_for_plot["CH4"] = DEFAULT_BANDS["CH4"]
        if "CO2" in selected_bands: bands_for_plot["CO2"] = DEFAULT_BANDS["CO2"]

        fig_st = plot_spectrum_interactive(ref_wl, np.nan_to_num(stacked_norm), fl_smooth=stacked_smooth, err=None, title="Stacked Spectrum", bands=bands_for_plot, show_bands_flag=show_bands and not raw_only, show_error=False)
        st.plotly_chart(fig_st, use_container_width=True, key=make_key('stacked','plot'))
        if show_snr and bands_for_plot:
            st.subheader("Stacked SNR (approx)")
            st.json({mol: float(np.round(calc_snr_on_band(ref_wl, stacked_smooth, rng),4)) for mol,rng in bands_for_plot.items()})
        if enable_downloads:
            df_stack = pd.DataFrame({"wavelength": ref_wl, "stacked": stacked_norm, "stacked_smoothed": stacked_smooth})
            # make the stacked download key unique to this tab
            dl_key = make_key('stacked','download','csv','stacked_tab')
            st.download_button("Download stacked CSV", df_stack.to_csv(index=False).encode('utf-8'), file_name="stacked_spectrum.csv", mime='text/csv', key=dl_key)

# Data Table tab
with tabs[4]:
    st.header("Data Table")
    for r in results:
        label = f"{r['file']} (HDU {r['hdu_index']})"
        st.subheader(label)
        df = pd.DataFrame({"wavelength": r['wl'], "flux": r['fl']})
        st.dataframe(df.head(500), use_container_width=True)
        if enable_downloads:
            dl_key = make_key(r['file'], r['hdu_index'], 'download', 'table_csv')
            st.download_button(f"Download CSV: {label}", df.to_csv(index=False).encode('utf-8'), file_name=f"{label}.csv", mime='text/csv', key=dl_key)

# Downloads tab
with tabs[5]:
    st.header("Downloads & Export")
    if enable_downloads:
        for r in results:
            label = f"{r['file']}_hdu{r['hdu_index']}"
            df = pd.DataFrame({"wavelength": r['wl'], "flux": r['fl']})
            dl_key = make_key(label, 'download', 'csv')
            st.download_button(f"CSV: {label}", df.to_csv(index=False).encode('utf-8'), file_name=f"{label}.csv", mime='text/csv', key=dl_key)

        if len(results) >= 2 and stack_enabled:
            min_wl = min(np.nanmin(r['wl']) for r in results)
            max_wl = max(np.nanmax(r['wl']) for r in results)
            ref_wl = np.linspace(min_wl, max_wl, 2000)
            interp_fluxes = [interp_to_reference(r['wl'], r['fl'], ref_wl) for r in results]
            arr = np.array(interp_fluxes)
            stacked = np.nanmedian(arr, axis=0) if stack_method=="median" else np.nanmean(arr, axis=0)
            if True:
                if np.nanmax(stacked) != np.nanmin(stacked):
                    stacked = (stacked - np.nanmin(stacked)) / (np.nanmax(stacked) - np.nanmin(stacked))
            df_stack = pd.DataFrame({"wavelength": ref_wl, "stacked": stacked})
            # unique key for the downloads tab stacked export
            dl_key = make_key('stacked','download','csv','downloads_tab')
            st.download_button("Download stacked CSV", df_stack.to_csv(index=False).encode('utf-8'), file_name="stacked_spectrum.csv", mime='text/csv', key=dl_key)
    else:
        st.info("Enable downloads in the sidebar to see export options.")

st.sidebar.success("Ready. Use the tabs to explore raw and processed data.")
st.caption("AstroFlow · FITSFlow MVP — upload data, toggle options, export results.")

# Images tab
with tabs[6]:
    st.header("FITS Images")
    found_image = False
    for r in results:
        # Open the FITS again for image HDUs
        with fits.open(r["path"], memmap=False) as hdul:
            for idx, hdu in enumerate(hdul):
                if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                    found_image = True
                    st.subheader(f"{r['file']} (HDU {idx}) — Image")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    im = ax.imshow(hdu.data, cmap="gray", origin="lower", aspect="auto")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)

                    # Export button
                    if enable_downloads:
                        import io
                        import PIL.Image as Image
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        st.download_button(
                            label=f"Download Image (PNG) — {r['file']} HDU {idx}",
                            data=buf,
                            file_name=f"{r['file']}_hdu{idx}_image.png",
                            mime="image/png",
                        )
                    plt.close(fig)

    if not found_image:
        st.info("No 2D images found in uploaded FITS files.")



# Reports tab
with tabs[7]:
    st.header("Generate PDF Report")
    st.markdown("Compile spectra, images, and tables into a single PDF.")

    if st.button("Generate Report"):
        import tempfile, os
        tmp_pdf = os.path.join(tempfile.gettempdir(), "astroflow_report.pdf")

        # Collect assets for the report
        plots = []
        images = []
        tables = []

        # Save example spectra plots as PNGs
        for res in results:
            wl, fl = res["wl"], res["fl"]
            fig = plot_spectrum_interactive(wl, fl, title=f"{res['file']} HDU {res['hdu_index']}")
            import io
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)
            img_path = os.path.join(tempfile.gettempdir(), f"{res['file']}_hdu{res['hdu_index']}_spectrum.png")
            with open(img_path, "wb") as f:
                f.write(buf.read())
            plots.append(img_path)

            # Save CSV for each
            df = pd.DataFrame({"wavelength": wl, "flux": fl})
            csv_path = os.path.join(tempfile.gettempdir(), f"{res['file']}_hdu{res['hdu_index']}.csv")
            df.to_csv(csv_path, index=False)
            tables.append(csv_path)

        # Collect 2D images
        for r in results:
            with fits.open(r["path"], memmap=False) as hdul:
                for idx, hdu in enumerate(hdul):
                    if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                        import matplotlib.pyplot as plt
                        img_path = os.path.join(tempfile.gettempdir(), f"{r['file']}_hdu{idx}_image.png")
                        plt.imsave(img_path, hdu.data, cmap="gray", origin="lower")
                        images.append(img_path)

        # Call reporter
        from FitsFlow.reporters import generate_pdf_report
        pdf_path = generate_pdf_report(
            output_path=tmp_pdf,
            metadata={"title": "AstroFlow Report"},
            plots=plots,
            tables=tables,
            images=images,
        )

        # Provide download
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="astroflow_report.pdf",
                mime="application/pdf",
            )

