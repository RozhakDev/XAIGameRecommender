import streamlit as st

from src.domain.entities import RecommendationItem

def render_recommendation_card(item: RecommendationItem) -> None:
    """
    Menampilkan kartu informasi lengkap untuk satu item game rekomendasi.

    Komponen ini menyajikan peringkat, judul game yang terhubung ke Steam,
    skor probabilitas prediksi, sebaran atensi antar-modalitas, dan alasan rekomendasi.

    Args:
        item (RecommendationItem): Entitas data game yang akan dirender.
    """
    dominant = item.attention.dominant_modality
    collab_w = item.attention.collaborative_pct
    tab_w = item.attention.tabular_pct
    txt_w = item.attention.textual_pct

    price_label = "Free" if item.price_final == 0.0 else f"${item.price_final:.2f}"
    discount_tag = f"<span style='background: #991b1b; color: #fef2f2; padding: 1px 5px; border-radius: 3px; font-size: 0.7rem; font-weight: 600;'>-{item.discount*100:.0f}%</span>" if item.discount > 0 else ""

    platforms = []
    if item.platforms.win:
        platforms.append("WIN")
    if item.platforms.mac:
        platforms.append("MAC")
    if item.platforms.linux:
        platforms.append("LINUX")
    if item.platforms.steam_deck:
        platforms.append("DECK")
    plat_str = " ".join([f"<span style='background: #1e293b; color: #cbd5e1; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem; font-family: monospace;'>{p}</span>" for p in platforms])

    card_html = (
        f"<div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 1.25rem 1.45rem; margin-bottom: 1.15rem; font-variant-numeric: tabular-nums;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem;'>"
        f"  <div style='display: flex; align-items: baseline; gap: 0.85rem; flex-wrap: wrap;'>"
        f"    <span style='font-family: monospace; font-size: 1.25rem; font-weight: 700; color: #38bdf8;'>{item.rank:02d}</span>"
        f"    <a href='https://store.steampowered.com/app/{item.app_id}' target='_blank' style='text-decoration: none; color: #f8fafc; font-size: 1.15rem; font-weight: 600;'>{item.title}</a>"
        f"    <span style='font-family: monospace; font-size: 0.72rem; color: #94a3b8; background: #080c14; padding: 2px 7px; border-radius: 4px; border: 1px solid #1e293b;'>APPID {item.app_id}</span>"
        f"  </div>"
        f"  <div style='text-align: right; min-width: 140px;'>"
        f"    <span style='font-family: monospace; font-size: 1.2rem; font-weight: 700; color: #10b981;'>{item.match_score_pct:.1f}%</span>"
        f"    <span style='font-size: 0.7rem; color: #94a3b8; display: block; font-family: monospace; letter-spacing: 0.05em;'>PREDICTED PROBABILITY</span>"
        f"  </div>"
        f"</div>"
        f"<div style='display: flex; flex-wrap: wrap; gap: 0.85rem; align-items: center; margin-bottom: 0.95rem; font-size: 0.83rem;'>"
        f"  <span style='color: #cbd5e1;'>Rating: <strong style='color: #10b981;'>{item.rating}</strong> ({item.positive_ratio*100:.0f}% Positive)</span>"
        f"  <span style='color: #475569;'>|</span>"
        f"  <span style='color: #cbd5e1;'>Price: <strong style='color: #f8fafc;'>{price_label}</strong> {discount_tag}</span>"
        f"  <span style='color: #475569;'>|</span>"
        f"  <div>{plat_str}</div>"
        f"</div>"
        f"<div style='margin-bottom: 0.95rem;'>"
        f"  <div style='display: flex; justify-content: space-between; font-size: 0.72rem; font-family: monospace; color: #94a3b8; margin-bottom: 0.4rem;'>"
        f"    <span>ATTENTION WEIGHT DISTRIBUTION</span>"
        f"    <span>"
        f"      <span style='color: #818cf8; font-weight: 600;'>Collaborative {collab_w:.1f}%</span> &nbsp;|&nbsp; "
        f"      <span style='color: #10b981; font-weight: 600;'>Metadata {tab_w:.1f}%</span> &nbsp;|&nbsp; "
        f"      <span style='color: #f59e0b; font-weight: 600;'>Semantics {txt_w:.1f}%</span>"
        f"    </span>"
        f"  </div>"
        f"  <div style='display: flex; height: 7px; border-radius: 4px; overflow: hidden; background: #080c14; border: 1px solid #1e293b; width: 100%;'>"
        f"    <div style='width: {collab_w}%; background: #818cf8;'></div>"
        f"    <div style='width: {tab_w}%; background: #10b981;'></div>"
        f"    <div style='width: {txt_w}%; background: #f59e0b;'></div>"
        f"  </div>"
        f"</div>"
        f"<div style='background: #080c14; border-left: 3px solid {dominant.hex_color}; padding: 0.75rem 1rem; border-radius: 0 4px 4px 0; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;'>"
        f"  <span style='font-family: monospace; font-size: 0.72rem; font-weight: 700; color: {dominant.hex_color}; text-transform: uppercase; margin-right: 0.6rem;'>[{dominant.tag_label}]</span>"
        f"  {item.explanation}"
        f"</div>"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
