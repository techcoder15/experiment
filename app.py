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

def detect
