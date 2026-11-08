import streamlit as st

from src.config import AppConfig
from src.domain.entities import BenchmarkPersona, InferenceResponse
from src.ui.components.card import render_recommendation_card
from src.ui.components.header import render_header
from src.ui.components.metrics import render_inference_metrics

def render_persona_tile(persona: BenchmarkPersona) -> None:
    """
    Menampilkan kartu ringkasan profil sampel beserta tombol pilih interaktif.

    Args:
        persona (BenchmarkPersona): Data profil sampel yang akan ditampilkan.
    """
    tile_html = (
        f"<div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 1.15rem 1.25rem; height: 165px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;'>"
        f"  <div>"
        f"    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;'>"
        f"      <span style='font-family: monospace; font-size: 0.72rem; color: #38bdf8; font-weight: 700;'>ID {persona.user_id}</span>"
        f"      <span style='background: #080c14; color: #94a3b8; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; border: 1px solid #1e293b;'>{persona.tag}</span>"
        f"    </div>"
        f"    <div style='font-weight: 600; font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.45rem;'>{persona.title}</div>"
        f"    <div style='color: #cbd5e1; font-size: 0.8rem; line-height: 1.45;'>{persona.description}</div>"
        f"  </div>"
        f"</div>"
    )
    st.markdown(tile_html, unsafe_allow_html=True)
    if st.button(f"Select {persona.title}", key=f"btn_audit_{persona.user_id}", width="stretch"):
        st.session_state.trigger_user_id = str(persona.user_id)
        st.rerun()

def render_layout(
    config: AppConfig,
    device_name: str,
    total_games: int,
    total_users: int,
    response: InferenceResponse | None,
    error_message: str | None
) -> None:
    """
    Mengatur tata letak utama halaman aplikasi secara menyeluruh.

    Fungsi ini mengoordinasikan render header telemetri, penanganan pesan error,
    tampilan profil sampel saat idle, serta feed rekomendasi setelah inferensi.

    Args:
        config (AppConfig): Konfigurasi aplikasi yang memuat pengaturan umum.
        device_name (str): Nama perangkat keras komputasi yang aktif.
        total_games (int): Total game dalam katalog evaluasi.
        total_users (int): Total pengguna dalam data latih.
        response (InferenceResponse | None): Hasil inferensi rekomendasi jika tersedia.
        error_message (str | None): Pesan kesalahan validasi input jika terjadi kekeliruan.
    """
    render_header(
        device_name=device_name,
        total_games=total_games,
        total_users=total_users,
        version=config.engine_version,
        title=config.app_title,
        subtitle=config.app_subtitle
    )

    if error_message:
        st.markdown(
            f"<div style='background: #450a0a; border: 1px solid #991b1b; color: #fecaca; "
            f"padding: 0.95rem 1.35rem; border-radius: 8px; font-family: monospace; font-size: 0.85rem; margin-bottom: 1.5rem;'>"
            f"<strong>VALIDATION ERROR:</strong> {error_message}"
            f"</div>",
            unsafe_allow_html=True
        )
        return

    if response is None:
        st.markdown(
            f"<div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 1.5rem 1.75rem; margin-bottom: 1.75rem;'>"
            f"<div style='font-family: monospace; font-size: 0.75rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;'>Workspace Ready</div>"
            f"<h4 style='margin: 0 0 0.5rem 0; color: #f8fafc; font-weight: 600; font-size: 1.25rem;'>Select a User Profile to Start</h4>"
            f"<p style='color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; max-width: 820px; margin: 0;'>"
            f"Explore game recommendations generated from collaborative interactions, game metadata, and title semantics. "
            f"Select a sample profile below or enter a specific Steam User ID in the sidebar to run the model."
            f"</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='font-family: monospace; font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.85rem;'>"
            f"Sample User Profiles"
            f"</div>",
            unsafe_allow_html=True
        )

        personas = config.benchmark_personas
        row1_cols = st.columns(3)
        for col, p in zip(row1_cols, personas[:3]):
            with col:
                render_persona_tile(p)

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        row2_cols = st.columns([1, 1, 1])
        for col, p in zip(row2_cols[:2], personas[3:5]):
            with col:
                render_persona_tile(p)
        return

    render_inference_metrics(response)

    feed_head_col1, feed_head_col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with feed_head_col1:
        st.markdown(
            f"<div style='font-family: monospace; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.85rem;'>"
            f"Top Recommendations ({len(response.recommendations)} Games)"
            f"</div>",
            unsafe_allow_html=True
        )
    with feed_head_col2:
        st.markdown(
            f"<div style='text-align: right; font-family: monospace; font-size: 0.75rem; color: #64748b; margin-bottom: 0.85rem;'>"
            f"Ranked by Predicted Probability"
            f"</div>",
            unsafe_allow_html=True
        )

    for item in response.recommendations:
        render_recommendation_card(item)
