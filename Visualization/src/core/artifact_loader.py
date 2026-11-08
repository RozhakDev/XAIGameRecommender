import zipfile
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import torch

from src.config import PathConfig
from src.model.architecture import MultiModalAttentionRecSys

class ArtifactLoadError(RuntimeError):
    """
    Pengecualian khusus untuk kesalahan saat memuat berkas artefak.

    Ditimbulkan ketika salah satu berkas model, kamus pemetaan, atau fitur
    tidak ditemukan atau gagal dibaca ke dalam memori.
    """
    pass

class ArtifactLoader:
    """
    Pengelola pemuatan model dan seluruh artefak serialisasi inferensi.

    Kelas ini memuat konfigurasi model, bobot terlatih PyTorch, pemetaan indeks,
    metadata game, serta tensor biner fitur ke perangkat komputasi target.
    """
    def __init__(self, paths: PathConfig = PathConfig(), device: torch.device | None = None):
        """
        Menyiapkan konfigurasi path dan perangkat komputasi loader.

        Args:
            paths (PathConfig, optional): Konfigurasi lokasi seluruh berkas artefak.
            device (torch.device, optional): Target perangkat komputasi CPU atau CUDA.
        """
        self.paths = paths
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_artifacts_exist()

    def _ensure_artifacts_exist(self) -> None:
        """
        Memeriksa keberadaan berkas artefak dan mengekstrak arsip jika diperlukan.

        Raises:
            ArtifactLoadError: Jika terdapat berkas artefak wajib yang hilang.
        """
        required_files = [
            self.paths.model_checkpoint,
            self.paths.model_config,
            self.paths.mappings,
            self.paths.games_metadata,
            self.paths.tabular_features,
            self.paths.textual_features
        ]
        missing = [f for f in required_files if not f.exists()]
        if missing and self.paths.artifacts_archive.exists():
            with zipfile.ZipFile(self.paths.artifacts_archive, "r") as archive:
                archive.extractall(self.paths.artifacts_dir)
        missing_after = [f for f in required_files if not f.exists()]
        if missing_after:
            missing_names = ", ".join(f.name for f in missing_after)
            raise ArtifactLoadError(f"Missing required artifact files: {missing_names}")

    def load_model_config(self) -> dict:
        """
        Memuat berkas konfigurasi arsitektur model dalam format JSON.

        Returns:
            dict: Kamus dimensi input dan hyperparameter arsitektur model.

        Raises:
            ArtifactLoadError: Jika berkas JSON gagal dibaca.
        """
        try:
            with open(self.paths.model_config, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load model config: {exc}") from exc

    def load_mappings(self) -> dict[str, dict]:
        """
        Memuat kamus pemetaan indeks pengguna dan katalog game.

        Returns:
            dict[str, dict]: Struktur kamus user2idx, game2idx, dan idx2game.

        Raises:
            ArtifactLoadError: Jika berkas pickle gagal didekode.
        """
        try:
            with open(self.paths.mappings, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load mappings: {exc}") from exc

    def load_games_metadata(self) -> pd.DataFrame:
        """
        Memuat tabel metadata lengkap katalog game Steam.

        Returns:
            pd.DataFrame: DataFrame metadata game berisi atribut harga, rating, dan platform.

        Raises:
            ArtifactLoadError: Jika berkas CSV metadata gagal dibaca.
        """
        try:
            df = pd.read_csv(self.paths.games_metadata)
            return df
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load games metadata: {exc}") from exc

    def load_tabular_tensor(self) -> torch.Tensor:
        """
        Memuat matriks fitur tabular metadata game ke dalam format PyTorch Tensor.

        Returns:
            torch.Tensor: Tensor fitur tabular numerik pada perangkat komputasi aktif.

        Raises:
            ArtifactLoadError: Jika berkas biner NumPy tabular gagal dimuat.
        """
        try:
            features = np.load(self.paths.tabular_features)
            return torch.tensor(features, dtype=torch.float32, device=self.device)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load tabular features: {exc}") from exc

    def load_textual_tensor(self) -> torch.Tensor:
        """
        Memuat matriks embedding teks judul game ke dalam format PyTorch Tensor.

        Returns:
            torch.Tensor: Tensor embedding teks berdimensi 384 pada perangkat komputasi aktif.

        Raises:
            ArtifactLoadError: Jika berkas biner NumPy teks gagal dimuat.
        """
        try:
            features = np.load(self.paths.textual_features)
            return torch.tensor(features, dtype=torch.float32, device=self.device)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load textual features: {exc}") from exc

    def load_model(self, config: dict) -> MultiModalAttentionRecSys:
        """
        Merekonstruksi arsitektur model dan memuat bobot terlatih dari checkpoint.

        Args:
            config (dict): Konfigurasi dimensi input dan hidden layer model.

        Returns:
            MultiModalAttentionRecSys: Model PyTorch yang siap digunakan untuk inferensi.

        Raises:
            ArtifactLoadError: Jika checkpoint bobot model gagal dimuat.
        """
        try:
            model = MultiModalAttentionRecSys(
                n_users=config["n_users"],
                n_games=config["n_games"],
                n_tab=config["n_tab"],
                n_txt=config["n_txt"],
                hidden_dim=config.get("hidden_dim", 64)
            )
            state_dict = torch.load(self.paths.model_checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load PyTorch model: {exc}") from exc
