import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import streamlit as st
import torch

from src.config import AppConfig
from src.core.artifact_loader import ArtifactLoader, ArtifactLoadError
from src.core.inference_engine import InferenceEngine, UserNotFoundError
from src.ui.components.sidebar import render_sidebar
from src.ui.layout import render_layout

st.set_page_config(
    page_title="XAIGameRecommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Initializing model and pre-computed tensors...")
def initialize_engine() -> tuple[InferenceEngine, str, int, int]:
    """
    Memuat dan menginisialisasi model rekomendasi beserta seluruh artefak data.

    Fungsi ini membaca konfigurasi, memuat bobot model PyTorch ke memori,
    serta menyiapkan kamus pemetaan dan tensor fitur untuk proses inferensi.

    Returns:
        tuple[InferenceEngine, str, int, int]: Objek mesin inferensi, nama perangkat aktif,
        total katalog game, dan total profil pengguna.

    Raises:
        ArtifactLoadError: Jika terjadi kendala saat memuat berkas artefak.
    """
    config = AppConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = ArtifactLoader(paths=config.paths, device=device)

    model_config = loader.load_model_config()
    mappings = loader.load_mappings()
    metadata_df = loader.load_games_metadata()
    tabular_tensor = loader.load_tabular_tensor()
    textual_tensor = loader.load_textual_tensor()
    model = loader.load_model(model_config)

    engine = InferenceEngine(
        model=model,
        user2idx=mappings["user2idx"],
        game2idx=mappings["game2idx"],
        idx2game=mappings["idx2game"],
        games_metadata=metadata_df,
        tabular_tensor=tabular_tensor,
        textual_tensor=textual_tensor,
        device=device
    )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_games = len(mappings["game2idx"])
    total_users = len(mappings["user2idx"])

    return engine, device_name, total_games, total_users

def main() -> None:
    """
    Menjalankan alur utama aplikasi antarmuka XAIGameRecommender.

    Mengelola status interaksi pengguna, memproses permintaan rekomendasi,
    serta menyajikan visualisasi data secara terpadu.
    """
    app_config = AppConfig()

    try:
        engine, device_name, total_games, total_users = initialize_engine()
    except ArtifactLoadError as err:
        st.error(
            f"Failed to initialize model artifacts: {err}. Verify artifact directory integrity.",
            title="System Initialization Error"
        )
        return

    user_id_input, top_k, submit_clicked = render_sidebar(app_config)

    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "current_error" not in st.session_state:
        st.session_state.current_error = None

    trigger_id = st.session_state.pop("trigger_user_id", None)
    target_to_run = trigger_id or (user_id_input if submit_clicked else None)

    if target_to_run:
        try:
            response = engine.predict(user_id=target_to_run, top_k=top_k)
            st.session_state.current_response = response
            st.session_state.current_error = None
        except UserNotFoundError as user_err:
            st.session_state.current_error = str(user_err)
            st.session_state.current_response = None
        except Exception as exc:
            st.session_state.current_error = f"Inference execution fault: {exc}"
            st.session_state.current_response = None

    render_layout(
        config=app_config,
        device_name=device_name,
        total_games=total_games,
        total_users=total_users,
        response=st.session_state.current_response,
        error_message=st.session_state.current_error
    )

if __name__ == "__main__":
    main()
