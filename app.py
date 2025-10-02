import streamlit as st
from streamlit_elements import elements, dashboard, mui, nivo
import lightkurve as lk
import pandas as pd
from fpdf import FPDF
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import base64  # For potential base64 if needed

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Embed tsParticles for space background (glowing stars and particles)
particles_html = """
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/tsparticles@3.5.0/tsparticles.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/tsparticles-preset-stars@3.0.0/tsparticles.preset.stars.bundle.min.js"></script>
    <style>
        #particles {
            width: 100%;
            height: 100%;
            background-color: #0a0a2a;
        }
    </style>
</head>
<body>
    <div id="particles"></div>
    <script>
        (async () => {
            await loadStarsPreset(tsParticles);
            await tsParticles.load("particles", {
                preset: "stars",
                background: {
                    color: "#0a0a2a"
                },
                particles: {
                    number: {
                        value: 200  // Subtle density
                    },
                    move: {
                        speed: 0.5  // Slow for space feel
                    },
                    opacity: {
                        value: { min: 0.1, max: 1 }  // Glowing effect
                    }
                }
            });
        })();
    </script>
</body>
</html>
"""
st.components.v1.html(particles_html, height=2000, width=2000)  # Large to cover

# CSS to fix particles as background
st.markdown("""
<style>
    iframe {
        position: fixed;
        left: 0;
        right: 0;
        top: 0;
        bottom: 0;
        z-index: -1;  /* Behind content */
    }
</style>
""", unsafe_allow_html=True)

# Session state for data persistence
if 'data' not in st.session_state:
    st.session_state.data = None
if 'mode' not in st.session_state:
    st.session_state.mode = "Raw TESS Mode (FITS)"
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'conclusion' not in st.session_state:
    st.session_state.conclusion = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Mode selector at top
st.session_state.mode = st.selectbox("Select Mode", ["Raw TESS Mode (FITS)", "TOI Mode (CSV)"], index=0 if st.session_state.mode == "Raw TESS Mode (FITS)" else 1)

# Upload slot
uploaded_file = st.file_uploader("Upload your file", type=["fits"] if "FITS" in st.session_state.mode else ["csv"])

# Status bar
status = st.empty()

if uploaded_file:
    status.info("Processing file...")
    try:
        if "FITS" in st.session_state.mode:
            # Raw TESS Mode
            lc = lk.LightCurve.read(uploaded_file, format='tess')
            lc = lc.normalize()
            flat_lc = lc.flatten()
            pg = flat_lc.to_periodogram(method='bls')
            period = pg.period_at_max_power.value
            t0 = pg.transit_time_at_max_power.btjd
            duration = pg.duration_at_max_power.value
            depth = pg.depth_at_max_power.value
            snr = pg.max_power.value  # Approximate SNR
            folded_lc = flat_lc.fold(period=period, epoch_time=t0)

            st.session_state.stats = {
                "Orbital Period (days)": f"{period:.2f}",
                "Transit Depth (ppm)": f"{depth * 1e6:.2f}",
                "Transit Duration (hours)": f"{duration * 24:.2f}",
                "SNR": f"{snr:.2f}"
            }
            st.session_state.conclusion = "Possible exoplanet detected" if snr > 10 else "No clear exoplanet signal"
            st.session_state.data = {
                "raw": [{"x": t.jd, "y": f} for t, f in zip(lc.time, lc.flux)],
                "cleaned": [{"x": t.jd, "y": f} for t, f in zip(flat_lc.time, flat_lc.flux)],
                "folded": [{"x": p, "y": f} for p, f in zip(folded_lc.phase, folded_lc.flux)]
            }
            st.session_state.lc = lc  # For report
            st.session_state.flat_lc = flat_lc
            st.session_state.folded_lc = folded_lc
            st.session_state.file_name = uploaded_file.name

        else:
            # TOI Mode
            df = pd.read_csv(uploaded_file)
            # Assume columns; adjust as needed
            st.session_state.df = df
            st.session_state.data = {
                "period_radius": [{"x": row['Period (days)'], "y": row['Planet Radius (R_Earth)']} for _, row in df.iterrows()],
                "depth_mag": [{"x": row['Stellar Magnitude'], "y": row['Depth (ppm)']} for _, row in df.iterrows()],
                # Density plot as scatter for simplicity
                "density": [{"x": row['Period (days)'], "y": row['Planet Radius (R_Earth)'], "size": 10} for _, row in df.iterrows()]  # Fake density
            }
            # Simple assessment
            st.session_state.conclusion = {row['TOI']: "Likely real" if row.get('Disposition', 'Candidate') == 'Candidate' and row['Depth (ppm)'] > 100 else "Unlikely" for _, row in df.iterrows()}
            st.session_state.file_name = uploaded_file.name

        status.success("Analysis complete!")
    except Exception as e:
        status.error(f"Error: {str(e)}")

# Dashboard if data available
if st.session_state.data:
    # Define draggable/resizable layout
    layout = [
        dashboard.Item("panel1", 0, 0, 4, 2, isResizable=True, isDraggable=True),  # e.g., Raw/Candidate Summary
        dashboard.Item("panel2", 4, 0, 4, 2),  # Graph 1
        dashboard.Item("panel3", 0, 2, 4, 2),  # Graph 2
        dashboard.Item("panel4", 4, 2, 4, 2),  # Graph 3 / Assessment
        dashboard.Item("panel5", 0, 4, 8, 1),  # Conclusion
    ]

    with elements("dashboard"):
        with dashboard.Grid(layout):
            # Common Nivo theme for dark space
            nivo_theme = {
                "background": "#0a0a2a",
                "textColor": "#ffffff",
                "grid": {"line": {"stroke": "#00ffff", "strokeWidth": 0.5}},
                "axis": {"ticks": {"text": {"fill": "#ffffff"}}}
            }

            if "FITS" in st.session_state.mode:
                # Panel 1: Detection Statistics Table
                with mui.Paper(key="panel1", sx={"padding": 2}):
                    mui.Typography("Detection Statistics", variant="h6")
                    with mui.Table():
                        with mui.TableHead():
                            mui.TableRow(mui.TableCell("Statistic"), mui.TableCell("Value"))
                        with mui.TableBody():
                            for k, v in st.session_state.stats.items():
                                mui.TableRow(mui.TableCell(k), mui.TableCell(v))

                # Panel 2: Raw Light Curve
                with mui.Paper(key="panel2", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Raw Light Curve", variant="h6")
                    nivo.Line(
                        data=[{"id": "Raw Flux", "data": st.session_state.data["raw"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear"},
                        yScale={"type": "linear"},
                        axisBottom={"legend": "Time (JD)", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Flux", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        enableSlices="x",
                        useMesh=True
                    )

                # Panel 3: Cleaned Light Curve
                with mui.Paper(key="panel3", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Cleaned Light Curve", variant="h6")
                    nivo.Line(
                        data=[{"id": "Cleaned Flux", "data": st.session_state.data["cleaned"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear"},
                        yScale={"type": "linear"},
                        axisBottom={"legend": "Time (JD)", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Flux", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        enableSlices="x",
                        useMesh=True
                    )

                # Panel 4: Phase-Folded Transit Curve
                with mui.Paper(key="panel4", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Phase-Folded Transit", variant="h6")
                    nivo.Line(
                        data=[{"id": "Folded Flux", "data": st.session_state.data["folded"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear"},
                        yScale={"type": "linear"},
                        axisBottom={"legend": "Phase", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Flux", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        enableSlices="x",
                        useMesh=True
                    )

            else:
                # TOI Mode Panels
                # Panel 1: Candidate Summary Table
                with mui.Paper(key="panel1", sx={"padding": 2}):
                    mui.Typography("Candidate Summary", variant="h6")
                    with mui.Table():
                        with mui.TableHead():
                            mui.TableRow(*[mui.TableCell(col) for col in st.session_state.df.columns])
                        with mui.TableBody():
                            for _, row in st.session_state.df.iterrows():
                                mui.TableRow(*[mui.TableCell(str(val)) for val in row])

                # Panel 2: Orbital Period vs Planet Radius
                with mui.Paper(key="panel2", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Period vs Radius", variant="h6")
                    nivo.ScatterPlot(
                        data=[{"id": "Candidates", "data": st.session_state.data["period_radius"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear", "min": "auto", "max": "auto"},
                        yScale={"type": "linear", "min": "auto", "max": "auto"},
                        axisBottom={"legend": "Period (days)", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Radius (R_Earth)", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        useMesh=True
                    )

                # Panel 3: Transit Depth vs Stellar Magnitude
                with mui.Paper(key="panel3", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Depth vs Magnitude", variant="h6")
                    nivo.ScatterPlot(
                        data=[{"id": "Candidates", "data": st.session_state.data["depth_mag"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear", "min": "auto", "max": "auto"},
                        yScale={"type": "linear", "min": "auto", "max": "auto"},
                        axisBottom={"legend": "Stellar Magnitude", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Depth (ppm)", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        useMesh=True
                    )

                # Panel 4: Candidate Density Plot (Scatter for clustering)
                with mui.Paper(key="panel4", sx={"height": "100%", "padding": 1}):
                    mui.Typography("Density Plot", variant="h6")
                    nivo.ScatterPlot(
                        data=[{"id": "Density", "data": st.session_state.data["density"]}],
                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                        xScale={"type": "linear"},
                        yScale={"type": "linear"},
                        nodeSize=10,
                        axisBottom={"legend": "Period (days)", "legendPosition": "middle", "legendOffset": 40},
                        axisLeft={"legend": "Radius (R_Earth)", "legendPosition": "middle", "legendOffset": -50},
                        theme=nivo_theme,
                        useMesh=True
                    )

            # Panel 5: Conclusion / Assessment
            with mui.Paper(key="panel5", sx={"padding": 2}):
                mui.Typography("Conclusion", variant="h6")
                if isinstance(st.session_state.conclusion, str):
                    mui.Typography(st.session_state.conclusion)
                else:
                    for toi, lik in st.session_state.conclusion.items():
                        mui.Typography(f"{toi}: {lik}")

# Reset button
if st.button("Reset / Clear Dashboard"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Export Report
if st.session_state.data and st.button("Download Full Report as PDF"):
    pdf = FPDF()
    pdf.set_fill_color(10, 10, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Metadata
    pdf.cell(200, 10, txt=f"File: {st.session_state.file_name}", ln=1)
    pdf.cell(200, 10, txt=f"Mode: {st.session_state.mode}", ln=1)

    # Add graphs using Matplotlib (for PDF export)
    if "FITS" in st.session_state.mode:
        # Raw LC
        fig, ax = plt.subplots(facecolor='#0a0a2a')
        st.session_state.lc.plot(ax=ax, color='#00ffff')
        ax.set_facecolor('#0a0a2a')
        ax.tick_params(colors='#ffffff')
        ax.set_xlabel('Time (JD)', color='#ffffff')
        ax.set_ylabel('Flux', color='#ffffff')
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        img_io.seek(0)
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)  # Space for image

        # Cleaned
        fig, ax = plt.subplots(facecolor='#0a0a2a')
        st.session_state.flat_lc.plot(ax=ax, color='#00ffff')
        ax.set_facecolor('#0a0a2a')
        ax.tick_params(colors='#ffffff')
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        img_io.seek(0)
        pdf.add_page()
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)

        # Folded
        fig, ax = plt.subplots(facecolor='#0a0a2a')
        st.session_state.folded_lc.plot(ax=ax, color='#00ffff')
        ax.set_facecolor('#0a0a2a')
        ax.tick_params(colors='#ffffff')
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        img_io.seek(0)
        pdf.add_page()
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)

        # Stats Table
        pdf.add_page()
        pdf.cell(200, 10, txt="Detection Statistics", ln=1)
        for k, v in st.session_state.stats.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=1)

    else:
        # TOI Graphs (use Plotly for export since scatter)
        import plotly.express as px
        # Period vs Radius
        fig = px.scatter(st.session_state.df, x='Period (days)', y='Planet Radius (R_Earth)', title='Period vs Radius')
        fig.update_layout(paper_bgcolor='#0a0a2a', plot_bgcolor='#0a0a2a', font_color='#ffffff')
        img_io = io.BytesIO()
        fig.write_image(img_io, format='png')
        img_io.seek(0)
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)

        # Depth vs Mag
        fig = px.scatter(st.session_state.df, x='Stellar Magnitude', y='Depth (ppm)', title='Depth vs Magnitude')
        fig.update_layout(paper_bgcolor='#0a0a2a', plot_bgcolor='#0a0a2a', font_color='#ffffff')
        img_io = io.BytesIO()
        fig.write_image(img_io, format='png')
        img_io.seek(0)
        pdf.add_page()
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)

        # Density (simple scatter)
        fig = px.scatter(st.session_state.df, x='Period (days)', y='Planet Radius (R_Earth)', title='Density Plot')
        fig.update_layout(paper_bgcolor='#0a0a2a', plot_bgcolor='#0a0a2a', font_color='#ffffff')
        img_io = io.BytesIO()
        fig.write_image(img_io, format='png')
        img_io.seek(0)
        pdf.add_page()
        pdf.image(img_io, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)

        # Summary Table
        pdf.add_page()
        pdf.cell(200, 10, txt="Candidate Summary", ln=1)
        for col in st.session_state.df.columns:
            pdf.cell(40, 10, txt=str(col), border=1)
        pdf.ln()
        for _, row in st.session_state.df.iterrows():
            for val in row:
                pdf.cell(40, 10, txt=str(val), border=1)
            pdf.ln()

    # Conclusion
    pdf.add_page()
    pdf.cell(200, 10, txt="Conclusion", ln=1)
    if isinstance(st.session_state.conclusion, str):
        pdf.multi_cell(200, 10, txt=st.session_state.conclusion)
    else:
        for toi, lik in st.session_state.conclusion.items():
            pdf.cell(200, 10, txt=f"{toi}: {lik}", ln=1)

    # Download
    pdf_output = pdf.output(dest='S')
    st.download_button(
        label="Download PDF",
        data=pdf_output.encode('latin-1'),
        file_name="exoplanet_report.pdf",
        mime="application/pdf"
    )
