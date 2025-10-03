import streamlit as st
import pandas as pd
import requests
from lightkurve import search_lightcurve, LightCurve
from astropy.timeseries import BoxLeastSquares
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use headless backend for server stability
import matplotlib.pyplot as plt
import urllib.parse
from sklearn.ensemble import IsolationForest 
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type 

# --------------------------------------
# CONFIGURATION
# --------------------------------------
MIN_PERIOD = 0.1
MAX_PERIOD = 30
BLS_DURATION = 0.1
BLS_POWER_THRESHOLD = 0.5
RMS_THRESHOLD = 0.0005
AMP_THRESHOLD = 0.001 
REL_STD_THRESHOLD = 0.001
ANOMALY_THRESHOLD = 0.05
PERIOD_TOLERANCE = 0.05

# --------------------------------------
# FUNCTION DEFINITIONS 
# --------------------------------------

def clean_lightcurve(lc: LightCurve) -> LightCurve:
    """Clean light curve using preferred flux columns without deprecation."""
    if 'pdcsap_flux' in lc.columns:
        flux_col = lc['pdcsap_flux']
        flux_err_col = lc['pdcsap_flux_err'] if 'pdcsap_flux_err' in lc.columns else None
    else:
        flux_col = lc['sap_flux']
        flux_err_col = lc['sap_flux_err'] if 'sap_flux_err' in lc.columns else None
        
    lc = lc.select_flux(flux_col, flux_err_col)
    
    # Simple cleaning: remove nans and 5-sigma outliers
    return lc.remove_nans().remove_outliers(sigma=5)

def compute_flux_stats(flux):
    """
    Compute flux statistics and ensure they are returned as native Python floats.
    
    This is the key fix for 'unhashable type: numpy.ndarray' when using cache,
    as numpy scalars can sometimes cause issues.
    """
    mean = np.mean(flux)
    std = np.std(flux)
    amp = (np.nanmax(flux) - np.nanmin(flux)) / mean
    rms = np.sqrt(np.mean((flux - mean)**2))
    
    # FIX: Explicitly convert numpy scalars to native Python floats
    return float(amp), float(rms), float(mean), float(std)

def classify_variable(power, threshold, amp, rms, rel_std):
    return (power > threshold and amp > AMP_THRESHOLD and
            rms > RMS_THRESHOLD and rel_std > REL_STD_THRESHOLD)

def detect_transit_candidates(flux, bls_folded, anomaly_threshold=ANOMALY_THRESHOLD):
    """Use Isolation Forest (ML) for anomaly detection on flux to flag potential transit dips."""
    try:
        # Normalize flux for ML
        flux_norm = (flux - np.mean(flux)) / np.std(flux)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination='auto', random_state=42) # 'auto' is safer than a fixed 0.1
        iso_forest.fit(flux_norm.reshape(-1, 1))
        anomalies = iso_forest.predict(flux_norm.reshape(-1, 1))
        
        num_anomalies = np.sum(anomalies == -1)
        anomaly_fraction = num_anomalies / len(flux)
        
        # Check folded light curve anomalies
        clustered_anomalies_fraction = 0.0
        if bls_folded is not None:
            folded_flux = bls_folded.flux.value
            folded_norm = (folded_flux - np.mean(folded_flux)) / np.std(folded_flux)
            folded_anomalies = iso_forest.predict(folded_norm.reshape(-1, 1))
            clustered_anomalies_fraction = np.sum(folded_anomalies == -1) / len(folded_flux)
        
        is_transit_candidate = (anomaly_fraction > anomaly_threshold) and (clustered_anomalies_fraction > anomaly_threshold)
        
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

@st.cache_data(ttl=3600) 
def get_exoplanet_info(tic_id):
    """Query NASA Exoplanet Archive for confirmed planets and their orbital periods, with retries."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _query_api():
        tic_clean = clean_tic_id(tic_id)
        # Using the standard TAP service query
        query = f"select pl_name, pl_orbper from ps where tic_id = '{tic_clean}'"
        encoded_query = urllib.parse.quote(query)
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status() 
        return response.json()

    try:
        data = _query_api()
        planets = []
        if data and isinstance(data, list):
            for row in data:
                # Robust parsing for JSON array of objects
                if isinstance(row, dict) and 'pl_name' in row and 'pl_orbper' in row:
                    name = row['pl_name']
                    period = row['pl_orbper']
                    period = float(period) if period and str(period).lower() not in ('nan', 'null') else None
                    if period:
                        planets.append((name, period))
        return len(planets) > 0, planets
    except Exception as e:
        # Warning only, don't stop analysis if exoplanet query fails
        st.warning(f"Exoplanet query failed after retries for {tic_id}: {str(e)}. Treating as non-host.")
        return False, [] 

# --------------------------------------
# CRITICAL FIX: CACHED DATA DOWNLOAD
# --------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _download_and_clean_lc(full_tic):
    """
    Downloads and cleans the Light Curve data, ensuring complex Lightkurve objects
    are NOT passed as arguments to other cached functions.
    """
    search_result = search_lightcurve(full_tic, author='TESS', exptime=120) # Prioritize 2-minute data
    if len(search_result) == 0:
        return None, None, None # lc_clean, lc_normalized, metadata
        
    selected_row = search_result[0]
    lc = selected_row.download()
    
    if lc is None:
        return None, None, None
        
    lc_clean = clean_lightcurve(lc)
    
    # ENHANCEMENT: Normalize the light curve for robust BLS analysis
    lc_normalized = lc_clean.normalize()
    
    # Extract metadata
    mission = selected_row.mission if hasattr(selected_row, 'mission') else 'Unknown'
    sector = selected_row.sector if hasattr(selected_row, 'sector') else 'Unknown'
    metadata = f"{mission} Sector {sector}"
    
    return lc_clean, lc_normalized, metadata


def run_analysis(tic_id, debug=False):
    """Run the full analysis for a single TIC ID and return results and figures."""
    try:
        full_tic = get_full_tic(tic_id)
        
        # 1. Exoplanet Info (Cached)
        is_exoplanet_host, planet_info = get_exoplanet_info(tic_id)
        
        # 2. Data Fetch and Cleaning (Cached)
        lc_clean, lc_normalized, metadata = _download_and_clean_lc(full_tic)

        if lc_clean is None:
            return None, f"No TESS data found for {full_tic} or download failed.", {}
            
        time = lc_normalized.time.value # Use normalized time for analysis
        flux = lc_normalized.flux.value # Use normalized flux for BLS and stats
        
        # 3. Flux Stats (Returns native floats)
        amp, rms, flux_mean, flux_std = compute_flux_stats(flux) 
        rel_std = flux_std / flux_mean
        display_amp = 100 * amp 

        if debug:
            st.info(f"🔍 Debug: Downloaded data from {metadata}")
            st.info(f"🔍 Debug: Flux stats computed - Amp: {display_amp:.2f}%, RMS: {rms:.6f}")

        # 4. BLS Analysis (on normalized flux)
        bls = BoxLeastSquares(time, flux)
        bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, 10000)
        bls_result = bls.power(bls_periods, BLS_DURATION)
        bls_best_period = bls_result.period[np.argmax(bls_result.power)]
        bls_max_power = np.max(bls_result.power)
        bls_variable = classify_variable(bls_max_power, BLS_POWER_THRESHOLD, amp, rms, rel_std) 

        # 5. Folded Light Curves (on normalized light curve)
        bls_folded = lc_normalized.fold(period=bls_best_period)

        # 6. ML Anomaly Detection (on normalized flux)
        is_transit_candidate, anomaly_fraction, num_anomalies, iso_forest = detect_transit_candidates(flux, bls_folded)

        # 7. Check period matching
        period_match = False
        if is_exoplanet_host and planet_info:
            detected_periods = [bls_best_period]
            for _, p_period in planet_info:
                for det_period in detected_periods:
                    if abs(det_period - p_period) / p_period < PERIOD_TOLERANCE:
                        period_match = True
                        break
                if period_match:
                    break

        # 8. Create Figures
        figs = {}
        
        # Cleaned/Normalized LC with anomalies
        fig_lc, ax_lc = plt.subplots()
        lc_normalized.plot(ax=ax_lc, title=f"Cleaned and Normalized Light Curve ({metadata})")
        if iso_forest is not None:
            # Re-normalize just for ML plotting consistency if needed, though flux is already normalized here
            anomalies_plot = iso_forest.predict(flux.reshape(-1, 1)) == -1 
            if np.sum(anomalies_plot) > 0:
                ax_lc.scatter(time[anomalies_plot], flux[anomalies_plot], color='red', s=1, alpha=0.5, label='ML Anomalies')
                ax_lc.legend()
        figs['lc'] = fig_lc
        
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
        
        # BLS Folded with anomalies
        fig_bls_fold, ax_bls_fold = plt.subplots()
        bls_folded.plot(ax=ax_bls_fold, title=f"BLS Folded Light Curve at {bls_best_period:.5f} days")
        if iso_forest is not None:
            # Re-run prediction on folded data for scatter plot
            folded_flux = bls_folded.flux.value
            # Use original fitted forest for consistency
            folded_anoms = iso_forest.predict(folded_flux.reshape(-1, 1)) == -1
            if np.sum(folded_anoms) > 0:
                ax_bls_fold.scatter(bls_folded.phase[folded_anoms], bls_folded.flux.value[folded_anoms], color='red', s=5, alpha=0.7, label='ML Anomalies')
                ax_bls_fold.legend()
        figs['bls_fold'] = fig_bls_fold

        # 9. Results and Classification
        known_planets = ', '.join([name for name, _ in planet_info]) if planet_info else 'None'
        results = {
            'Data Source': metadata, # NEW
            'BLS Best Period': f"{bls_best_period:.5f} days",
            'BLS Max Power': f"{bls_max_power:.5f}",
            'Amplitude': f"{display_amp:.2f}%",
            'RMS': f"{rms:.6f}",
            'Relative Std Dev': f"{rel_std:.6f}",
            'Confirmed Exoplanet Host': 'Yes' if is_exoplanet_host else 'No',
            'Known Planets': known_planets,
            'Period Matches Known Planet': 'Yes' if period_match else 'No',
            'ML Transit Candidate': 'Yes' if is_transit_candidate else 'No',
            'Anomaly Fraction': f"{anomaly_fraction:.3f}"
        }
        
        classification = ""
        if is_exoplanet_host:
            classification = "🌌 Confirmed Exoplanet Host Star (NASA Archive)"
            if period_match:
                classification += " with **matching detected period**"
            if is_transit_candidate:
                classification += " and ML-detected transit-like anomalies"
            if bls_variable:
                classification += ", showing stellar/planetary variability"
        elif is_transit_candidate:
            classification = "🪐 **ML Transit Candidate** (anomalies suggest potential exoplanet signal)"
            if bls_variable:
                classification += " + variability confirmed by BLS"
        elif bls_variable:
            classification = "⚠️ **Stellar Variability** detected by BLS/Metrics only"
        else:
            classification = "❌ No strong variability detected"
        
        return results, classification, figs
        
    except ValueError as ve:
        return None, str(ve), {}
    except Exception as e:
        # Catch all other exceptions, log them, and return a general error
        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        return None, f"Error: {str(e)}", {}

# --------------------------------------
# STREAMLIT APP
# --------------------------------------
st.title("TESS Light Curve Variability Analyzer with AI/ML Integration 🔭")
st.set_page_config(layout="wide", page_title="TESS Analyzer")

st.sidebar.header("Input Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode (Show Analysis Steps)", value=False)
input_mode = st.sidebar.radio("Choose input method:", ("Single TIC ID", "Upload CSV"))

if input_mode == "Single TIC ID":
    tic_id = st.sidebar.text_input("Enter TIC ID (e.g., TIC 168789840):", value="TIC 168789840")
    if st.sidebar.button("Analyze"):
        if tic_id:
            with st.spinner("Analyzing light curve data with BLS and ML..."):
                results, classification, figs = run_analysis(tic_id, debug=debug_mode)
            
            if results:
                st.header(f"Results for {tic_id}")
                st.markdown(f"**Classification:** {classification}")
                
                col_m, col_p = st.columns([1, 2])
                
                with col_m:
                    st.subheader("Key Metrics")
                    for key, value in results.items():
                        st.write(f"**{key}:** {value}")
                
                with col_p:
                    st.subheader("Plots")
                    # Display plots in a consistent layout
                    if 'lc' in figs:
                        st.pyplot(figs['lc'])
                        plt.close(figs['lc'])
                    
                    col_bls1, col_bls2 = st.columns(2)
                    with col_bls1:
                         if 'bls_period' in figs:
                            st.pyplot(figs['bls_period'])
                            plt.close(figs['bls_period'])
                    with col_bls2:
                        if 'bls_fold' in figs:
                            st.pyplot(figs['bls_fold'])
                            plt.close(figs['bls_fold'])
            else:
                st.error(f"Analysis failed: {classification}")
        else:
            st.warning("Please enter a TIC ID.")

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with TIC IDs (one per row, no header):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        raw_ids = df[0].tolist()
        tic_ids = []
        for raw_id in raw_ids:
            try:
                cleaned = clean_tic_id(raw_id)
                tic_ids.append(f"TIC {cleaned}")
            except ValueError:
                st.warning(f"Skipping invalid TIC ID: {raw_id}")
        st.sidebar.write(f"Loaded {len(tic_ids)} valid TIC IDs from {len(raw_ids)} entries.")
        
        if st.sidebar.button("Analyze All"):
            all_results = []
            
            progress_bar = st.progress(0)
            
            for i, full_tic in enumerate(tic_ids):
                progress_bar.progress((i + 1) / len(tic_ids), text=f"Analyzing {full_tic} ({i+1}/{len(tic_ids)})")
                
                with st.expander(f"{full_tic} ({i+1}/{len(tic_ids)})", expanded=False):
                    results, classification, figs = run_analysis(full_tic, debug=debug_mode)
                    
                    if results:
                        results['TIC ID'] = full_tic
                        results['Classification Summary'] = classification
                        all_results.append(results)
                        
                        st.markdown(f"**Classification:** {classification}")
                        st.subheader("Key Metrics")
                        st.dataframe(pd.DataFrame(results.items(), columns=['Metric', 'Value']).set_index('Metric'))
                        
                        st.subheader("Plots")
                        for key, fig in figs.items():
                            try:
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Plot '{key}' failed: {e}")
                    else:
                        st.error(f"Analysis failed for {full_tic}: {classification}")
            
            progress_bar.empty()
            st.success("Batch analysis complete!")
            
            if all_results:
                final_df = pd.DataFrame(all_results).set_index('TIC ID')
                st.subheader("Batch Summary Table")
                st.dataframe(final_df)
                
                # Download button for CSV
                csv = final_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Full Results CSV",
                    data=csv,
                    file_name='tess_analysis_results.csv',
                    mime='text/csv',
                )
    else:
        st.info("Upload a CSV file with TIC IDs to get started.")

# Example note
st.sidebar.markdown("---")
st.sidebar.info("✨ **Enhancements:**\n- **Bug Fixes:** Robust `numpy` type conversion and isolated caching.\n- **Normalization:** BLS/Analysis runs on **normalized flux** for scientific accuracy.\n- **Data Source:** Added data source to results.\n\n**Example Confirmed Exoplanet Host:** Try TIC 261136679")
