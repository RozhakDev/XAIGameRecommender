import streamlit as st

def render_header(
    device_name: str,
    total_games: int,
    total_users: int,
    version: str,
    title: str = "XAIGameRecommender",
    subtitle: str = "Explainable Steam Recommendation Engine • Multi-Modal Attention Attribution"
) -> None:
    """
    Menampilkan bilah navigasi atas dan ringkasan telemetri sistem.

    Komponen ini menyajikan identitas aplikasi, status kesiapan komputasi,
    serta total katalog dan profil pengguna yang terdaftar.

    Args:
        device_name (str): Nama perangkat komputasi yang sedang aktif (CPU atau CUDA).
        total_games (int): Jumlah keseluruhan game unik dalam katalog evaluasi.
        total_users (int): Jumlah keseluruhan profil pengguna dalam pemetaan latih.
        version (str): Nomor rilis versi aplikasi saat ini.
        title (str): Nama judul platform aplikasi.
        subtitle (str): Deskripsi fungsi sistem.
    """
    head_col, status_col = st.columns([0.72, 0.28], vertical_alignment="center")
    with head_col:
        st.markdown(
            f"<div style='display: flex; align-items: baseline; gap: 0.75rem;'>"
            f"<span style='font-size: 1.45rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;'>{title}</span>"
            f"<span style='font-family: monospace; font-size: 0.72rem; color: #38bdf8; background: #0f172a; padding: 2px 7px; border-radius: 4px; border: 1px solid #1e293b;'>{version}</span>"
            f"</div>"
            f"<div style='font-size: 0.82rem; color: #94a3b8; margin-top: 0.15rem;'>"
            f"{subtitle}"
            f"</div>",
            unsafe_allow_html=True
        )
    with status_col:
        st.markdown(
            f"<div style='text-align: right; font-family: monospace; font-size: 0.78rem; color: #10b981; background: #0f172a; padding: 0.45rem 0.85rem; border-radius: 6px; border: 1px solid #1e293b; display: inline-block; float: right;'>"
            f"Compute: {device_name.upper()} | Status: Ready"
            f"</div>",
            unsafe_allow_html=True
        )

    telemetry_html = (
        f"<div style='display: flex; flex-wrap: wrap; gap: 1.5rem; font-family: monospace; font-size: 0.78rem; "
        f"color: #94a3b8; padding: 0.55rem 0; margin-top: 0.75rem; margin-bottom: 1.25rem; border-top: 1px solid #1e293b; border-bottom: 1px solid #1e293b;'>"
        f"<span>Dataset: <strong style='color: #f8fafc;'>{total_games:,}</strong> Games</span>"
        f"<span>Profiles: <strong style='color: #f8fafc;'>{total_users:,}</strong> Users</span>"
        f"<span>Architecture: <strong style='color: #f8fafc;'>MultiModalAttentionRecSys (64-dim)</strong></span>"
        f"</div>"
    )
    st.markdown(telemetry_html, unsafe_allow_html=True)
