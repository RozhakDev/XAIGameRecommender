from dataclasses import dataclass

from src.domain.enums import ModalityType

@dataclass(frozen=True)
class BenchmarkPersona:
    """
    Data profil pengguna sampel untuk pengujian rekomendasi.

    Kelas ini memuat identitas pengguna, judul persona, label ringkas,
    dan penjelasan pola bermain untuk kebutuhan audit cepat.
    """
    user_id: int
    title: str
    tag: str
    description: str

@dataclass(frozen=True)
class PlatformSupport:
    """
    Informasi ketersediaan platform sistem operasi game.

    Menyimpan status kompatibilitas game pada platform Windows,
    macOS, Linux, dan perangkat Steam Deck.
    """
    win: bool
    mac: bool
    linux: bool
    steam_deck: bool

@dataclass(frozen=True)
class AttentionDistribution:
    """
    Sebaran bobot atensi antar-modalitas hasil komputasi model.

    Menyimpan porsi perhatian model terhadap fitur kolaboratif, metadata
    tabular, dan teks judul saat menghasilkan prediksi.
    """
    collaborative: float
    tabular: float
    textual: float

    @property
    def dominant_modality(self) -> ModalityType:
        """
        Menentukan modalitas yang memiliki bobot atensi tertinggi.

        Returns:
            ModalityType: Jenis modalitas yang paling dominan mempengaruhi prediksi.
        """
        scores = [self.collaborative, self.tabular, self.textual]
        max_idx = scores.index(max(scores))
        return ModalityType(max_idx)

    @property
    def collaborative_pct(self) -> float:
        """
        Mengonversi bobot atensi kolaboratif ke dalam bentuk persentase.

        Returns:
            float: Nilai persentase atensi kolaboratif dalam rentang 0 hingga 100.
        """
        return self.collaborative * 100.0

    @property
    def tabular_pct(self) -> float:
        """
        Mengonversi bobot atensi metadata tabular ke dalam bentuk persentase.

        Returns:
            float: Nilai persentase atensi tabular dalam rentang 0 hingga 100.
        """
        return self.tabular * 100.0

    @property
    def textual_pct(self) -> float:
        """
        Mengonversi bobot atensi teks semantik ke dalam bentuk persentase.

        Returns:
            float: Nilai persentase atensi teks dalam rentang 0 hingga 100.
        """
        return self.textual * 100.0

@dataclass(frozen=True)
class RecommendationItem:
    """
    Informasi lengkap satu judul game hasil rekomendasi.

    Kelas ini menggabungkan atribut metadata katalog game, skor kecocokan,
    sebaran atensi modalitas, serta narasi alasan rekomendasi.
    """
    rank: int
    game_idx: int
    app_id: int
    title: str
    match_score: float
    rating: str
    price_original: float
    price_final: float
    discount: float
    positive_ratio: float
    platforms: PlatformSupport
    attention: AttentionDistribution
    explanation: str

    @property
    def match_score_pct(self) -> float:
        """
        Mengonversi skor probabilitas prediksi model menjadi format persentase.

        Returns:
            float: Nilai kecocokan rekomendasi dalam satuan persen.
        """
        return self.match_score * 100.0

@dataclass(frozen=True)
class InferenceResponse:
    """
    Wadah hasil eksekusi inferensi rekomendasi secara menyeluruh.

    Menyimpan informasi pengguna, daftar rekomendasi teratas, durasi waktu komputasi,
    dan total ruang katalog game yang berhasil dievaluasi.
    """
    user_id: int
    user_idx: int
    top_k: int
    recommendations: list[RecommendationItem]
    latency_seconds: float
    total_candidates_scored: int

    @property
    def dominant_bias(self) -> ModalityType:
        """
        Menentukan modalitas yang paling sering dominan pada seluruh hasil rekomendasi.

        Returns:
            ModalityType: Modalitas utama yang menjadi konsensus faktor rekomendasi.
        """
        if not self.recommendations:
            return ModalityType.TABULAR
        counts = {m: 0 for m in ModalityType}
        for rec in self.recommendations:
            counts[rec.attention.dominant_modality] += 1
        return max(counts, key=counts.get)
