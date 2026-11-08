from src.domain.entities import AttentionDistribution
from src.domain.enums import ModalityType

class ReasoningExplainer:
    """
    Penerjemah bobot atensi model menjadi narasi penjelasan yang ramah.

    Kelas ini memetakan distribusi atensi tertinggi ke dalam kalimat penjelasan
    yang objektif dan berbasis data bagi pengguna akhir.
    """
    def explain(self, attention: AttentionDistribution, rating: str, price: float) -> str:
        """
        Menyusun narasi alasan rekomendasi berdasarkan modalitas yang paling dominan.

        Args:
            attention (AttentionDistribution): Sebaran bobot atensi antar-modalitas.
            rating (str): Label rating ulasan komunitas game.
            price (float): Nilai harga game dalam satuan dolar.

        Returns:
            str: Kalimat penjelasan alasan rekomendasi yang mudah dipahami.
        """
        dominant = attention.dominant_modality
        if dominant == ModalityType.COLLABORATIVE:
            return "Collaborative Factor: Recommendation is driven by player interaction patterns similar to users who positively reviewed this game."
        if dominant == ModalityType.TABULAR:
            formatted_price = f"{price:.2f}" if isinstance(price, (int, float)) else str(price)
            return f"Metadata Factor: Recommendation is primarily driven by game attributes, aligning with titles rated '{rating}' and priced around ${formatted_price}."
        return "Semantic Factor: Recommendation is driven by thematic text similarity in game titles and descriptive keywords."
