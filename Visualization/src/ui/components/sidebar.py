import streamlit as st

from src.config import AppConfig

def render_sidebar(config: AppConfig) -> tuple[str, int, bool]:
    """
    Menampilkan panel kontrol samping untuk kueri rekomendasi.

    Fungsi ini menyediakan pilihan profil sampel atau input ID manual, slider jumlah
    rekomendasi, tombol eksekusi, dan ringkasan arsitektur.

    Args:
        config (AppConfig): Konfigurasi aplikasi yang memuat persona dan batasan nilai.

    Returns:
        tuple[str, int, bool]: Kumpulan nilai (user_id_string, nilai_top_k,
        status_tombol_ditekan).
    """
    with st.sidebar:
        st.markdown(
            f"<div style='margin-bottom: 0.25rem;'>"
            f"<span style='font-family: monospace; font-size: 0.78rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase;'>Recommendation Query</span>"
            f"<div style='font-size: 0.78rem; color: #94a3b8; margin-top: 0.15rem;'>Configure Input & Ranking Parameters</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='margin-bottom: 0.85rem;'></div>", unsafe_allow_html=True)

        mode = st.radio(
            "Profile Source Mode",
            options=["Sample Profile", "Custom User ID"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

        user_id_val = ""
        if mode == "Sample Profile":
            persona_labels = [f"{p.title} ({p.tag})" for p in config.benchmark_personas]
            selected_idx = st.selectbox(
                "Select Sample Profile",
                options=range(len(persona_labels)),
                format_func=lambda i: persona_labels[i],
                index=0,
                label_visibility="collapsed"
            )
            active_persona = config.benchmark_personas[selected_idx]
            user_id_val = str(active_persona.user_id)
            st.markdown(
                f"<div style='font-family: monospace; font-size: 0.75rem; color: #94a3b8; background: #0f172a; padding: 0.35rem 0.65rem; border-radius: 4px; border: 1px solid #1e293b; margin-top: 0.25rem; margin-bottom: 0.5rem;'>"
                f"Selected ID: <strong style='color: #f8fafc;'>{user_id_val}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<span style='font-size: 0.75rem; font-family: monospace; color: #94a3b8;'>ENTER STEAM USER ID</span>", unsafe_allow_html=True)
            user_id_val = st.text_input(
                "Custom Steam User ID",
                value="",
                placeholder="e.g. 2967327",
                label_visibility="collapsed"
            ).strip()

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.75rem; font-family: monospace; color: #94a3b8;'>RESULTS COUNT (TOP-K)</span>", unsafe_allow_html=True)
        top_k = st.slider(
            "Top-K Results",
            min_value=config.min_top_k,
            max_value=config.max_top_k,
            value=config.default_top_k,
            step=1,
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-bottom: 0.85rem;'></div>", unsafe_allow_html=True)
        submit_btn = st.button(
            "Get Recommendations",
            type="primary",
            width="stretch"
        )

        st.markdown("<div style='margin-top: 2rem; border-top: 1px solid #1e293b; padding-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-family: monospace; font-size: 0.72rem; color: #64748b; line-height: 1.6;'>"
            f"<div style='color: #94a3b8; font-weight: 700; margin-bottom: 0.25rem;'>MODEL SPECIFICATIONS</div>"
            f"<div>Architecture: MultiModalAttentionRecSys</div>"
            f"<div>Latent Representation: 64 Units</div>"
            f"<div>Text Embeddings: all-MiniLM-L6-v2</div>"
            f"<div>Trained Parameters: 124.7M</div>"
            f"<div>Engine Release: {config.engine_version}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        return user_id_val, top_k, submit_btn
