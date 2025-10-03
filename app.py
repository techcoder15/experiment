from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
