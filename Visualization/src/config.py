from dataclasses import dataclass
from pathlib import Path

from src.domain.entities import BenchmarkPersona

@dataclass(frozen=True)
class PathConfig:
    """
    Menyimpan konfigurasi path berkas dan direktori artefak.

    Kelas ini memusatkan seluruh lokasi file model, pemetaan indeks,
    dan fitur yang dibutuhkan selama proses inferensi.
    """
    base_dir: Path = Path(__file__).resolve().parent.parent
    artifacts_dir: Path = base_dir / "artifacts"
    model_checkpoint: Path = artifacts_dir / "best_recsys_model.pth"
    model_config: Path = artifacts_dir / "model_config.json"
    mappings: Path = artifacts_dir / "mappings.pkl"
    games_metadata: Path = artifacts_dir / "games_metadata.csv"
    tabular_features: Path = artifacts_dir / "tabular_features.npy"
    textual_features: Path = artifacts_dir / "textual_features.npy"
    artifacts_archive: Path = artifacts_dir / "artifacts.zip"

@dataclass(frozen=True)
class AppConfig:
    """
    Mengelola pengaturan umum dan parameter antarmuka aplikasi.

    Kelas ini memuat informasi identitas aplikasi, batasan jumlah rekomendasi,
    serta profil sampel untuk memudahkan evaluasi pengguna.
    """
    paths: PathConfig = PathConfig()
    app_title: str = "XAIGameRecommender"
    app_subtitle: str = "Explainable Steam Recommendation Engine • Multi-Modal Attention Attribution"
    engine_version: str = "v2.4"
    default_top_k: int = 5
    min_top_k: int = 1
    max_top_k: int = 50
    benchmark_personas: tuple[BenchmarkPersona, ...] = (
        BenchmarkPersona(
            user_id=2967327,
            title="User 2967327",
            tag="51 Reviews",
            description="Active interaction history across indie, action, and adventure titles."
        ),
        BenchmarkPersona(
            user_id=9417270,
            title="User 9417270",
            tag="38 Reviews",
            description="Consistent engagement pattern with strategy and simulation titles."
        ),
        BenchmarkPersona(
            user_id=1729562,
            title="User 1729562",
            tag="29 Reviews",
            description="Preference pattern centered on community-acclaimed releases."
        ),
        BenchmarkPersona(
            user_id=6220080,
            title="User 6220080",
            tag="24 Reviews",
            description="Interaction record focused on fast-paced action and arcade gameplay."
        ),
        BenchmarkPersona(
            user_id=847558,
            title="User 847558",
            tag="18 Reviews",
            description="Multi-platform catalog history including Steam Deck verified titles."
        )
    )
