import streamlit as st
import pandas as pd
from lightkurve import search_lightcurve
from astropy.timeseries import LombScargle, BoxLeastSquares
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --------------------------------------
# CONFIGURATION (same as core)
# --------------------------------------
MIN_PERIOD = 0.1
MAX_PERIOD = 30
BLS_DURATION = 0.1
FAP_LEVEL = 0.01
BLS_POWER_THRESHOLD = 0.5
RMS_THRESHOLD = 0.0005
AMP_THRESHOLD = 0.5
REL_STD_THRESHOLD = 0.001

# --------------------------------------
# FUNCTION DEFINITIONS (same as core, adapted for Streamlit)
# --------------------------------------

def clean_lightcurve(lc):
    if hasattr(lc, 'PDCSAP_FLUX') and lc.PDCSAP_FLUX is not None:
        lc = lc.PDCSAP_FLUX
    else:
        lc = lc.SAP_FLUX
    return lc.remove_nans().remove_outliers()

def compute_flux_stats(flux):
    mean = np.mean(flux)
    std = np.std(flux)
    amp = 100 * (np.nanmax(flux) - np.nanmin(flux)) / mean
    rms = np.sqrt(np.mean((flux - mean)**2))
    return amp, rms, mean, std

def classify_variable(power, threshold, amp, rms, rel_std):
    return (power > threshold and amp > AMP_THRESHOLD and
            rms > RMS_THRESHOLD and rel_std > REL_STD_THRESHOLD)

def run_analysis(tic_id):
    """Run the full analysis for a single TIC ID and return results and figures."""
    try:
        # Data Fetch and Cleaning
        sector_data = search_lightcurve(tic_id)
        if len(sector_data) == 0:
            return None, "No data found for TIC ID."
        lc = sector_data[1].download()
        lc_clean = clean_lightcurve(lc)
        time = lc_clean.time.value
        flux = lc_clean.flux.value
        amp, rms, flux_mean, flux_std = compute_flux_stats(flux)
        rel_std = flux_std / flux_mean

        # Lomb-Scargle Analysis
        lk_periodogram = lc_clean.to_periodogram(method="lombscargle", minimum_period=MIN_PERIOD, maximum_period=MAX_PERIOD)
        ls_best_period = lk_periodogram.period_at_max_power
        ls_max_power = lk_periodogram.max_power
        ls = LombScargle(time, flux)
        ls_freq, ls_power = ls.autopower(minimum_frequency=1/MAX_PERIOD, maximum_frequency=1/MIN_PERIOD)
        fap = ls.false_alarm_level(FAP_LEVEL)
        ls_variable = classify_variable(ls_max_power.value if hasattr(ls_max_power, "value") else ls_max_power,
                                        fap, amp, rms, rel_std)

        # BLS Analysis
        bls = BoxLeastSquares(time, flux)
        bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, 10000)
        bls_result = bls.power(bls_periods, BLS_DURATION)
        bls_best_period = bls_result.period[np.argmax(bls_result.power)]
        bls_max_power = np.max(bls_result.power)
        bls_variable = classify_variable(bls_max_power, BLS_POWER_THRESHOLD, amp, rms, rel_std)

        # Folded Light Curves
        ls_folded = lc_clean.fold(period=ls_best_period)
        bls_folded = lc_clean.fold(period=bls_best_period)

        # Create figures
        figs = {}
        
        # Cleaned LC
        fig_lc, ax_lc = plt.subplots()
        lc_clean.plot(ax=ax_lc, title="Cleaned Light Curve")
        figs['lc'] = fig_lc
        
        # LS Periodogram
        fig_ls, ax_ls = plt.subplots()
        lk_periodogram.plot(ax=ax_ls, title="Lomb-Scargle Periodogram")
        ax_ls.axhline(fap, color='red', linestyle='--', label='1% FAP Threshold')
        ax_ls.legend()
        figs['ls_period'] = fig_ls
        
        # BLS Periodogram
        fig_bls, ax_bls = plt.subplots()
        ax_bls.plot(bls_result.period, bls_result.power, color='purple')
        ax_bls.axvline(bls_best_period, color='green', linestyle='--', label=f"Best Period: {bls_best_period:.5f} d")
        ax_bls.set_xlabel("Period [days]")
        ax_bls.set_ylabel("BLS Power")
        ax_bls.set_title("Box Least Squares Periodogram")
        ax_bls.legend()
        ax_bls.grid(True)
        figs['bls_period'] = fig_bls
        
        # LS Folded
        fig_ls_fold, ax_ls_fold = plt.subplots()
        ls_folded.plot(ax=ax_ls_fold, title=f"LS Folded Light Curve at {ls_best_period.value:.5f} days")
        figs['ls_fold'] = fig_ls_fold
        
        # BLS Folded
        fig_bls_fold, ax_bls_fold = plt.subplots()
        bls_folded.plot(ax=ax_bls_fold, title=f"BLS Folded Light Curve at {bls_best_period:.5f} days")
        figs['bls_fold'] = fig_bls_fold

        # Results dict
        results = {
            'LS Best Period': f"{ls_best_period.value:.5f} days",
            'LS Max Power': f"{ls_max_power:.5f}",
            'BLS Best Period': f"{bls_best_period:.5f} days",
            'BLS Max Power': f"{bls_max_power:.5f}",
            '1% FAP (LS)': f"{fap:.5f}",
            'Amplitude': f"{amp:.2f}%",
            'RMS': f"{rms:.6f}",
            'Relative Std Dev': f"{rel_std:.6f}"
        }
        
        classification = ""
        if ls_variable and bls_variable:
            classification = "✅ Confirmed VARIABLE STAR by both LS and BLS"
        elif ls_variable:
            classification = "⚠️ Variability detected by Lomb-Scargle only"
        elif bls_variable:
            classification = "⚠️ Variability detected by BLS only"
        else:
            classification = "❌ No strong variability detected"
        
        return results, classification, figs
    except Exception as e:
        return None, f"Error: {str(e)}", {}

# --------------------------------------
# STREAMLIT APP
# --------------------------------------
st.title("TESS Light Curve Variability Analyzer")

# Sidebar for input options
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input method:", ("Single TIC ID", "Upload CSV"))

if input_mode == "Single TIC ID":
    tic_id = st.sidebar.text_input("Enter TIC ID (e.g., TIC 168789840):", value="TIC 168789840")
    if st.sidebar.button("Analyze"):
        if tic_id:
            with st.spinner("Analyzing..."):
                results, classification, figs = run_analysis(tic_id)
                if results:
                    st.header(f"Results for {tic_id}")
                    st.subheader("Key Metrics")
                    for key, value in results.items():
                        st.write(f"**{key}:** {value}")
                    st.subheader("Classification")
                    st.write(classification)
                    
                    st.subheader("Plots")
                    for key, fig in figs.items():
                        st.pyplot(fig)
                        plt.close(fig)  # Close to free memory
                else:
                    st.error(classification)
        else:
            st.warning("Please enter a TIC ID.")

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with TIC IDs (one per row, no header):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        tic_ids = df[0].tolist()  # Assuming first column is TIC IDs
        st.sidebar.write(f"Loaded {len(tic_ids)} TIC IDs.")
        
        if st.sidebar.button("Analyze All"):
            for i, tic_id in enumerate(tic_ids):
                with st.expander(f"TIC ID: {tic_id} ({i+1}/{len(tic_ids)})", expanded=False):
                    with st.spinner(f"Analyzing {tic_id}..."):
                        results, classification, figs = run_analysis(tic_id)
                        if results:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Key Metrics")
                                for key, value in results.items():
                                    st.write(f"**{key}:** {value}")
                            with col2:
                                st.subheader("Classification")
                                st.write(classification)
                            
                            st.subheader("Plots")
                            for key, fig in figs.items():
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.error(classification)
    else:
        st.info("Upload a CSV file with TIC IDs to get started.")
