import time

import pandas as pd
import numpy as np

import torch

from src.domain.entities import (
    AttentionDistribution,
    InferenceResponse,
    PlatformSupport,
    RecommendationItem
)
from src.core.explainer import ReasoningExplainer
from src.model.architecture import MultiModalAttentionRecSys

class UserNotFoundError(ValueError):
    """
    Pengecualian ketika ID pengguna tidak ditemukan dalam katalog latih.

    Ditimbulkan saat pengguna memasukkan identitas yang belum pernah
    memiliki riwayat interaksi dalam dataset.
    """
    pass

class InferenceEngine:
    """
    Mesin inferensi utama untuk menghasilkan rekomendasi game secara cepat.

    Kelas ini memvalidasi identitas pengguna, melakukan komputasi skor probabilitas
    pada seluruh katalog game secara paralel, dan mengurutkan hasil terbaik.
    """
    def __init__(
        self,
        model: MultiModalAttentionRecSys,
        user2idx: dict,
        game2idx: dict,
        idx2game: dict,
        games_metadata: pd.DataFrame,
        tabular_tensor: torch.Tensor,
        textual_tensor: torch.Tensor,
        explainer: ReasoningExplainer | None = None,
        device: torch.device | None = None
    ):
        """
        Menginisialisasi dependensi model, indeks pemetaan, dan matriks fitur inferensi.

        Args:
            model (MultiModalAttentionRecSys): Model PyTorch yang telah dimuat bobotnya.
            user2idx (dict): Kamus pemetaan dari user_id asli ke indeks integer model.
            game2idx (dict): Kamus pemetaan dari app_id asli ke indeks integer model.
            idx2game (dict): Kamus pemetaan balik dari indeks model ke app_id asli.
            games_metadata (pd.DataFrame): Dataframe berisi detail informasi game.
            tabular_tensor (torch.Tensor): Matriks fitur tabular yang telah dialokasikan.
            textual_tensor (torch.Tensor): Matriks fitur teks yang telah dialokasikan.
            explainer (ReasoningExplainer, optional): Modul penyusun alasan rekomendasi.
            device (torch.device, optional): Perangkat komputasi CPU atau CUDA.
        """
        self.model = model
        self.user2idx = user2idx
        self.game2idx = game2idx
        self.idx2game = idx2game
        self.tabular_tensor = tabular_tensor
        self.textual_tensor = textual_tensor
        self.explainer = explainer or ReasoningExplainer()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_games_tensor = torch.arange(len(self.game2idx), dtype=torch.long, device=self.device)
        self.metadata_lookup = self._build_metadata_lookup(games_metadata)

    def _build_metadata_lookup(self, df: pd.DataFrame) -> dict[int, dict]:
        """
        Membangun kamus lookup metadata game berbasis app_id untuk akses instan.

        Args:
            df (pd.DataFrame): Dataframe metadata katalog game Steam.

        Returns:
            dict[int, dict]: Kamus pemetaan app_id ke baris atribut game.
        """
        records = df.to_dict(orient="records")
        return {int(rec["app_id"]): rec for rec in records}

    def resolve_user_idx(self, raw_user_id: int | str) -> tuple[int, int]:
        """
        Memvalidasi dan menerjemahkan input ID pengguna ke indeks integer internal.

        Args:
            raw_user_id (int | str): Identitas pengguna mentah dari form input.

        Returns:
            tuple[int, int]: Pasangan (user_id_integer, user_idx_internal).

        Raises:
            UserNotFoundError: Jika format ID tidak valid atau tidak ada dalam data latih.
        """
        try:
            parsed_id = int(raw_user_id)
        except (ValueError, TypeError) as exc:
            raise UserNotFoundError(f"User ID '{raw_user_id}' harus berupa nilai integer valid.") from exc

        if parsed_id in self.user2idx:
            return parsed_id, self.user2idx[parsed_id]

        np_parsed = np.int64(parsed_id)
        if np_parsed in self.user2idx:
            return parsed_id, self.user2idx[np_parsed]

        raise UserNotFoundError(f"User ID {parsed_id} tidak ditemukan dalam data latih.")

    def predict(self, user_id: int | str, top_k: int = 5) -> InferenceResponse:
        """
        Menjalankan prediksi probabilitas dan ekstraksi Top-K game terbaik.

        Args:
            user_id (int | str): Identitas pengguna target rekomendasi.
            top_k (int, optional): Jumlah game teratas yang ingin ditampilkan.

        Returns:
            InferenceResponse: Objek hasil rekomendasi lengkap beserta metrik komputasi.
        """
        resolved_id, u_idx = self.resolve_user_idx(user_id)
        candidate_count = len(self.game2idx)
        clamped_k = min(max(1, top_k), candidate_count)

        start_time = time.perf_counter()

        u_tensor = torch.full((candidate_count,), u_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds, atts = self.model(
                u_tensor,
                self.all_games_tensor,
                self.tabular_tensor,
                self.textual_tensor,
                return_attention=True
            )

        top_scores, top_indices = torch.topk(preds, k=clamped_k)
        elapsed_seconds = time.perf_counter() - start_time

        top_scores_np = top_scores.cpu().numpy()
        top_indices_np = top_indices.cpu().numpy()
        top_atts_np = atts[top_indices].cpu().numpy()

        items: list[RecommendationItem] = []
        for rank, (score, g_idx, att_weights) in enumerate(zip(top_scores_np, top_indices_np, top_atts_np), start=1):
            app_id = int(self.idx2game[g_idx])
            meta = self.metadata_lookup.get(app_id, {})

            title = str(meta.get("title", f"Game App ID {app_id}"))
            rating = str(meta.get("rating", "Unknown"))
            price_orig = float(meta.get("price_original", 0.0))
            price_final = float(meta.get("price_final", 0.0))
            discount = float(meta.get("discount", 0.0))
            positive_ratio = float(meta.get("positive_ratio", 0.0))

            platforms = PlatformSupport(
                win=bool(meta.get("win", 1)),
                mac=bool(meta.get("mac", 0)),
                linux=bool(meta.get("linux", 0)),
                steam_deck=bool(meta.get("steam_deck", 0))
            )

            attention = AttentionDistribution(
                collaborative=float(att_weights[0]),
                tabular=float(att_weights[1]),
                textual=float(att_weights[2])
            )

            explanation = self.explainer.explain(
                attention=attention,
                rating=rating,
                price=price_orig
            )

            item = RecommendationItem(
                rank=rank,
                game_idx=int(g_idx),
                app_id=app_id,
                title=title,
                match_score=float(score),
                rating=rating,
                price_original=price_orig,
                price_final=price_final,
                discount=discount,
                positive_ratio=positive_ratio,
                platforms=platforms,
                attention=attention,
                explanation=explanation
            )
            items.append(item)

        return InferenceResponse(
            user_id=resolved_id,
            user_idx=u_idx,
            top_k=clamped_k,
            recommendations=items,
            latency_seconds=elapsed_seconds,
            total_candidates_scored=candidate_count
        )
