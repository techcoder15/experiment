import streamlit as st
import pandas as pd
import requests
from lightkurve import search_lightcurve
from astropy.timeseries import LombScargle, BoxLeastSquares
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend for server stability
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import urllib.parse  # For quoting the query
from sklearn.ensemble import IsolationForest  # ML integration for anomaly detection
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # New: For API retries

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
ANOMALY_THRESHOLD = 0.05  # Fraction of points flagged as anomalies to suggest potential transit
PERIOD_TOLERANCE = 0.05  # 5% relative tolerance for period matching

# --------------------------------------
# FUNCTION DEFINITIONS (same as core, adapted for Streamlit)
# --------------------------------------

def clean_lightcurve(lc):
    """Clean light curve using preferred flux columns without deprecation."""
    # Prefer pdcsap_flux if available, else sap_flux
    if 'pdcsap_flux' in lc.columns:
        flux_col = lc['pdcsap_flux']
        flux_err_col = lc['pdcsap_flux_err'] if 'pdcsap_flux_err' in lc.columns else None
    else:
        flux_col = lc['sap_flux']
        flux_err_col = lc['sap_flux_err'] if 'sap_flux_err' in lc.columns else None
    # Select the flux column
    lc = lc.select_flux(flux_col, flux_err_col)
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

def detect_transit_candidates(flux, bls_folded, anomaly_threshold=ANOMALY_THRESHOLD):
    """Use Isolation Forest (ML) for anomaly detection on flux to flag potential transit dips."""
    try:
        # Normalize flux
        flux_norm = (flux - np.mean(flux)) / np.std(flux)
        
        # Fit Isolation Forest on full flux
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(flux_norm.reshape(-1, 1))
        anomalies = iso_forest.predict(flux_norm.reshape(-1, 1))
        
        # Count anomalies (negative labels are anomalies)
        num_anomalies = np.sum(anomalies == -1)
        anomaly_fraction = num_anomalies / len(flux)
        
        # Check folded if provided: predict (no refit)
        clustered_anomalies = 0.0
        if bls_folded is not None:
            folded_flux = bls_folded.flux.value
            folded_norm = (folded_flux - np.mean(folded_flux)) / np.std(folded_flux)
            folded_anomalies = iso_forest.predict(folded_norm.reshape(-1, 1))
            clustered_anomalies = np.sum(folded_anomalies == -1) / len(folded_flux)
        
        is_transit_candidate = (anomaly_fraction > anomaly_threshold) and (clustered_anomalies > anomaly_threshold)
        
        return is_transit_candidate, anomaly_fraction, num_anomalies, iso_forest
    except Exception as e:
        st.warning(f"ML anomaly detection failed: {str(e)}")
        return False, 0.0, 0, None

def clean_tic_id(tic_id):
    """Clean TIC ID: remove 'TIC ' prefix, strip, and validate."""
    tic_clean = str(tic_id).replace('TIC ', '').replace('TIC', '').strip()
    if not tic_clean.isdigit():
        raise ValueError(f"Invalid TIC ID: {tic_id}")
    return tic_clean

def get_full_tic(tic_id):
    """Get full TIC string for lightkurve search."""
    tic_num = clean_tic_id(tic_id)
    return f"TIC {tic_num}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_exoplanet_info(tic_id):
    """Query NASA Exoplanet Archive for confirmed planets and their orbital periods, with retries."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _query_api():
        tic_clean = clean_tic_id(tic_id)
        query = f"select pl_name, pl_orbper from ps where tic_id = '{tic_clean}'"
        encoded_query = urllib.parse.quote(query)
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=json"
        response = requests.get(url, timeout=30)  # Increased timeout
        response.raise_for_status()  # Raise if not 200
        return response.json()

    try:
        data = _query_api()
        if data and 'result' in data and data['result'] and 'table' in data['result'] and 'data' in data['result']['table']:
            planets = []
            for row in data['result']['table']['data']:
                name = row[0] if row[0] else 'Unknown'
                period_str = row[1]
                period = float(period_str) if period_str and str(period_str).lower() != 'nan' else None
                if period:
                    planets.append((name, period))
            return len(planets) > 0, planets
        return False, []
    except Exception as e:
        st.warning(f"Exoplanet query failed after retries for {tic_id}: {str(e)}. Treating as non-host.")
        return False, []  # Fallback: Assume no planets to avoid false negatives

def run_analysis(tic_id, debug=False):
    """Run the full analysis for a single TIC ID and return results and figures."""
    try:
        if debug:
            st.info(f"🔍 Debug: Starting analysis for {tic_id}")
        
        full_tic = get_full_tic(tic_id)
        tic_clean = clean_tic_id(tic_id)
        
        if debug:
            st.info(f"🔍 Debug: Cleaned TIC: {tic_clean}")
        
        # Check for confirmed exoplanet info
        if debug:
            st.info("🔍 Debug: Querying exoplanet archive...")
        is_exoplanet_host, planet_info = get_exoplanet_info(tic_id)
        
        if debug:
            st.info(f"🔍 Debug: Exoplanet host: {is_exoplanet_host}, Planets: {len(planet_info)}")
        
        # Data Fetch and Cleaning
        if debug:
            st.info("🔍 Debug: Fetching light curve data...")
        sector_data = search_lightcurve(full_tic)
        if len(sector_data) == 0:
            return None, "No data found for TIC ID.", {}
        # Use the first sector instead of second to avoid index errors
        selected_row = sector_data[0]
        lc = selected_row.download()
        if debug:
            mission = selected_row.mission if hasattr(selected_row, 'mission') else 'Unknown'
            sector = selected_row.sector if hasattr(selected_row, 'sector') else mission.split(' ') [-1] if 'Sector' in mission else 'Unknown'
            st.info(f"🔍 Debug: Downloaded LC from {mission} Sector {sector}")
        
        if debug:
            st.info("🔍 Debug: Cleaning light curve...")
        lc_clean = clean_lightcurve(lc)
        time = lc_clean.time.value
        flux = lc_clean.flux.value
        if debug:
            st.info(f"🔍 Debug: LC cleaned - {len(time)} points")
        
        amp, rms, flux_mean, flux_std = compute_flux_stats(flux)
        rel_std = flux_std / flux_mean
        if debug:
            st.info(f"🔍 Debug: Flux stats computed - Amp: {amp:.2f}%, RMS: {rms:.6f}")

        # Lomb-Scargle Analysis
        if debug:
            st.info("🔍 Debug: Running Lomb-Scargle...")
        lk_periodogram = lc_clean.to_periodogram(method="lombscargle", minimum_period=MIN_PERIOD, maximum_period=MAX_PERIOD)
        ls_best_period = lk_periodogram.period_at_max_power
        ls_max_power = lk_periodogram.max_power
        ls = LombScargle(time, flux)
        ls_freq, ls_power = ls.autopower(minimum_frequency=1/MAX_PERIOD, maximum_frequency=1/MIN_PERIOD)
        fap = ls.false_alarm_level(FAP_LEVEL)
        ls_variable = classify_variable(ls_max_power.value if hasattr(ls_max_power, "value") else ls_max_power,
                                        fap, amp, rms, rel_std)
        if debug:
            st.info(f"🔍 Debug: LS complete - Best period: {ls_best_period.value:.5f}, Variable: {ls_variable}")

        # BLS Analysis
        if debug:
            st.info("🔍 Debug: Running BLS...")
        bls = BoxLeastSquares(time, flux)
        bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, 10000)
        bls_result = bls.power(bls_periods, BLS_DURATION)
        bls_best_period = bls_result.period[np.argmax(bls_result.power)]
        bls_max_power = np.max(bls_result.power)
        bls_variable = classify_variable(bls_max_power, BLS_POWER_THRESHOLD, amp, rms, rel_std)
        if debug:
            st.info(f"🔍 Debug: BLS complete - Best period: {bls_best_period:.5f}, Variable: {bls_variable}")

        # Folded Light Curves
        ls_folded = lc_clean.fold(period=ls_best_period)
        bls_folded = lc_clean.fold(period=bls_best_period)
        if debug:
            st.info("🔍 Debug: Folded light curves created")

        # ML Anomaly Detection for Transit Candidates (after BLS)
        if debug:
            st.info("🔍 Debug: Running ML anomaly detection...")
        is_transit_candidate, anomaly_fraction, num_anomalies, iso_forest = detect_transit_candidates(flux, bls_folded)
        if debug:
            st.info(f"🔍 Debug: ML complete - Transit candidate: {is_transit_candidate}, Anomalies: {num_anomalies}")

        # Check period matching for known planets
        period_match = False
        if is_exoplanet_host and planet_info:
            detected_periods = [ls_best_period.value, bls_best_period]
            for _, p_period in planet_info:
                for det_period in detected_periods:
                    if abs(det_period - p_period) / p_period < PERIOD_TOLERANCE:
                        period_match = True
                        break
                if period_match:
                    break
            if debug:
                st.info(f"🔍 Debug: Period match: {period_match}")

        if debug:
            st.info("🔍 Debug: Creating plots...")

        # Create figures
        figs = {}
        
        # Cleaned LC with anomalies highlighted (ML viz)
        fig_lc, ax_lc = plt.subplots()
        lc_clean.plot(ax=ax_lc, title="Cleaned Light Curve with ML Anomalies")
        if iso_forest is not None:
            flux_norm_plot = ((flux - flux_mean) / flux_std).reshape(-1, 1)
            anomalies_plot = iso_forest.predict(flux_norm_plot) == -1
            if np.sum(anomalies_plot) > 0:
                ax_lc.scatter(time[anomalies_plot], flux[anomalies_plot], color='red', s=1, alpha=0.5, label='ML Anomalies')
                ax_lc.legend()
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
        
        # BLS Folded with anomalies
        fig_bls_fold, ax_bls_fold = plt.subplots()
        bls_folded.plot(ax=ax_bls_fold, title=f"BLS Folded Light Curve at {bls_best_period:.5f} days with ML Anomalies")
        if iso_forest is not None:
            folded_flux_norm = ((bls_folded.flux.value - np.mean(bls_folded.flux.value)) / np.std(bls_folded.flux.value)).reshape(-1, 1)
            folded_anoms = iso_forest.predict(folded_flux_norm) == -1
            if np.sum(folded_anoms) > 0:
                ax_bls_fold.scatter(bls_folded.phase[folded_anoms], bls_folded.flux.value[folded_anoms], color='red', s=5, alpha=0.7, label='ML Anomalies')
                ax_bls_fold.legend()
        figs['bls_fold'] = fig_bls_fold

        if debug:
            st.info("🔍 Debug: Analysis complete - Displaying results")

        # Results dict
        known_planets = ', '.join([name for name, _ in planet_info]) if planet_info else 'None'
        results = {
            'LS Best Period': f"{ls_best_period.value:.5f} days",
            'LS Max Power': f"{ls_max_power:.5f}",
            'BLS Best Period': f"{bls_best_period:.5f} days",
            'BLS Max Power': f"{bls_max_power:.5f}",
            '1% FAP (LS)': f"{fap:.5f}",
            'Amplitude': f"{amp:.2f}%",
            'RMS': f"{rms:.6f}",
            'Relative Std Dev': f"{rel_std:.6f}",
            'Confirmed Exoplanet Host': 'Yes' if is_exoplanet_host else 'No',
            'Known Planets': known_planets,
            'Period Matches Known Planet': 'Yes' if period_match else 'No',
            'ML Transit Candidate': 'Yes' if is_transit_candidate else 'No',
            'Anomaly Fraction': f"{anomaly_fraction:.3f}"
        }
        
        # Updated Classification with ML and Period Matching
        classification = ""
        if is_exoplanet_host:
            classification = "🌌 Confirmed Exoplanet Host Star (from NASA Exoplanet Archive)"
            if period_match:
                classification += " with detected period matching known planet orbit"
            if is_transit_candidate:
                classification += " with ML-detected transit-like anomalies"
            if ls_variable or bls_variable:
                classification += " showing stellar/planetary variability"
        elif is_transit_candidate:
            classification = "🪐 ML Transit Candidate (anomalies suggest potential exoplanet signal)"
            if ls_variable or bls_variable:
                classification += " + variability confirmed by LS/BLS"
        elif ls_variable and bls_variable:
            classification = "✅ Confirmed VARIABLE STAR by both LS and BLS"
        elif ls_variable:
            classification = "⚠️ Variability detected by Lomb-Scargle only"
        elif bls_variable:
            classification = "⚠️ Variability detected by BLS only"
        else:
            classification = "❌ No strong variability detected"
        
        return results, classification, figs
    except ValueError as ve:
        return None, str(ve), {}
    except Exception as e:
        return None, f"Error: {str(e)}", {}

# --------------------------------------
# STREAMLIT APP
# --------------------------------------
st.title("TESS Light Curve Variability Analyzer with ai/ML Integration")
st.set_page_config(layout="wide", page_title="TESS Analyzer")

# Sidebar for input options
st.sidebar.header("Input Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode (Show Analysis Steps)", value=False)
input_mode = st.sidebar.radio("Choose input method:", ("Single TIC ID", "Upload CSV"))

if input_mode == "Single TIC ID":
    tic_id = st.sidebar.text_input("Enter TIC ID (e.g., TIC 168789840):", value="TIC 168789840")
    if st.sidebar.button("Analyze"):
        if tic_id:
            with st.spinner("Analyzing with AI/ML..."):
                results, classification, figs = run_analysis(tic_id, debug=debug_mode)
                if results:
                    st.header(f"Results for {tic_id}")
                    st.subheader("Key Metrics")
                    for key, value in results.items():
                        st.write(f"**{key}:** {value}")
                    st.subheader("Classification")
                    st.write(classification)
                    
                    st.subheader("Plots")
                    for key, fig in figs.items():
                        try:
                            st.pyplot(fig)
                            plt.close(fig)  # Close to free memory
                        except Exception as e:
                            st.error(f"Plot '{key}' failed: {e}")
                else:
                    st.error(classification)
        else:
            st.warning("Please enter a TIC ID.")

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with TIC IDs (one per row, no header):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        # Clean and filter valid TIC IDs
        raw_ids = df[0].tolist()
        tic_ids = []
        for raw_id in raw_ids:
            try:
                cleaned = clean_tic_id(raw_id)
                tic_ids.append(f"TIC {cleaned}")  # Store as full for analysis
            except ValueError:
                st.warning(f"Skipping invalid TIC ID: {raw_id}")
        st.sidebar.write(f"Loaded {len(tic_ids)} valid TIC IDs from {len(raw_ids)} entries.")
        
        if st.sidebar.button("Analyze All"):
            for i, full_tic in enumerate(tic_ids):
                with st.expander(f"{full_tic} ({i+1}/{len(tic_ids)})", expanded=False):
                    with st.spinner(f"Analyzing {full_tic} with AI/ML..."):
                        results, classification, figs = run_analysis(full_tic, debug=debug_mode)
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
                                try:
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.error(f"Plot '{key}' failed: {e}")
                        else:
                            st.error(classification)
    else:
        st.info("Upload a CSV file with TIC IDs to get started.")

# Example note
st.sidebar.markdown("---")
st.sidebar.info("**Example Confirmed Exoplanet Host:** Try TIC 261136679 (Pi Mensae, host of TESS-discovered planet Pi Men c)\n\n**ML Feature:** Isolation Forest detects anomalous flux dips as potential transits.\n\n**New:** Period matching against known exoplanet orbits for better classification.\n\n**Tip:** Ensure CSV has clean TIC IDs (e.g., 123456789, no 'TIC ID' labels).")
