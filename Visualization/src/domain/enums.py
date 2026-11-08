from enum import IntEnum

class ModalityType(IntEnum):
    """
    Representasi jenis modalitas data pada model rekomendasi.

    Enum ini membedakan ketiga sumber informasi utama, yaitu interaksi pemain,
    metadata tabular, dan semantik teks judul game.
    """
    COLLABORATIVE = 0
    TABULAR = 1
    TEXTUAL = 2

    @property
    def display_name(self) -> str:
        """
        Nama lengkap modalitas untuk ditampilkan pada antarmuka.

        Returns:
            str: Label deskriptif modalitas dalam bahasa yang ramah.
        """
        mapping = {
            ModalityType.COLLABORATIVE: "Collaborative Interaction",
            ModalityType.TABULAR: "Game Metadata (Rating & Price)",
            ModalityType.TEXTUAL: "Title Text Semantics"
        }
        return mapping[self]

    @property
    def tag_label(self) -> str:
        """
        Label singkat modalitas untuk penanda visual pada kartu rekomendasi.

        Returns:
            str: Tag ringkas huruf kapital untuk badge antarmuka.
        """
        mapping = {
            ModalityType.COLLABORATIVE: "COLLABORATIVE",
            ModalityType.TABULAR: "METADATA",
            ModalityType.TEXTUAL: "SEMANTICS"
        }
        return mapping[self]

    @property
    def hex_color(self) -> str:
        """
        Kode warna heksadesimal representasi visual modalitas.

        Returns:
            str: Nilai warna CSS untuk memperindah tampilan grafik dan border.
        """
        mapping = {
            ModalityType.COLLABORATIVE: "#818cf8",
            ModalityType.TABULAR: "#10b981",
            ModalityType.TEXTUAL: "#f59e0b"
        }
        return mapping[self]
