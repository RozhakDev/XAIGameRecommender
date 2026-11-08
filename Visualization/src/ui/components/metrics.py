import streamlit as st

from src.domain.entities import InferenceResponse

def render_inference_metrics(response: InferenceResponse) -> None:
    """
    Menampilkan deretan metrik performa utama hasil inferensi model.

    Komponen ini menyajikan ID pengguna, durasi komputasi dalam milidetik,
    total game yang diskor, serta faktor modalitas atensi paling dominan.

    Args:
        response (InferenceResponse): Objek respons inferensi yang memuat data metrik.
    """
    latency_ms = response.latency_seconds * 1000.0
    throughput = (response.total_candidates_scored / response.latency_seconds) if response.latency_seconds > 0 else 0
    bias = response.dominant_bias

    hud_html = (
        f"<div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem 1.25rem; "
        f"margin-bottom: 1.5rem; display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; font-variant-numeric: tabular-nums;'>"
        f"<div>"
        f"  <div style='font-size: 0.72rem; font-family: monospace; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Target User</div>"
        f"  <div style='font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 0.15rem;'>{response.user_id}</div>"
        f"  <div style='font-size: 0.75rem; color: #64748b; font-family: monospace;'>Dataset Index #{response.user_idx}</div>"
        f"</div>"
        f"<div>"
        f"  <div style='font-size: 0.72rem; font-family: monospace; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Inference Time</div>"
        f"  <div style='font-size: 1.25rem; font-weight: 700; color: #38bdf8; margin-top: 0.15rem;'>{latency_ms:.1f} ms</div>"
        f"  <div style='font-size: 0.75rem; color: #64748b; font-family: monospace;'>{throughput:,.0f} games/sec</div>"
        f"</div>"
        f"<div>"
        f"  <div style='font-size: 0.72rem; font-family: monospace; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Catalog Evaluated</div>"
        f"  <div style='font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 0.15rem;'>{response.total_candidates_scored:,}</div>"
        f"  <div style='font-size: 0.75rem; color: #64748b; font-family: monospace;'>Complete Catalog</div>"
        f"</div>"
        f"<div>"
        f"  <div style='font-size: 0.72rem; font-family: monospace; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Primary Factor</div>"
        f"  <div style='font-size: 1.25rem; font-weight: 700; color: {bias.hex_color}; margin-top: 0.15rem;'>{bias.tag_label}</div>"
        f"  <div style='font-size: 0.75rem; color: #64748b; font-family: monospace;'>Highest Mean Attention</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(hud_html, unsafe_allow_html=True)
