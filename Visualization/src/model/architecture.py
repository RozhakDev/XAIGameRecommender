import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAttentionRecSys(nn.Module):
    """
    Arsitektur neural network rekomendasi berbasis atensi multi-modal.

    Model ini memadukan embedding kolaboratif pengguna dan game dengan fitur tabular
    serta teks semantik melalui mekanisme atensi terkondisi profil pemain.
    """
    def __init__(self, n_users: int, n_games: int, n_tab: int, n_txt: int, hidden_dim: int = 64):
        """
        Menginisialisasi seluruh lapisan embedding, proyeksi, dan atensi model.

        Args:
            n_users (int): Total jumlah pengguna unik dalam katalog latih.
            n_games (int): Total jumlah game unik dalam katalog latih.
            n_tab (int): Dimensi vektor fitur tabular metadata game.
            n_txt (int): Dimensi vektor embedding teks judul game.
            hidden_dim (int, optional): Ukuran dimensi ruang representasi laten bersama.
        """
        super().__init__()
        self.u_emb = nn.Embedding(n_users, hidden_dim)
        self.g_emb = nn.Embedding(n_games, hidden_dim)
        self.tab_proj = nn.Sequential(nn.Linear(n_tab, hidden_dim), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.Linear(n_txt, hidden_dim), nn.ReLU())

        self.q_layer = nn.Linear(hidden_dim, hidden_dim)
        self.k_layer = nn.Linear(hidden_dim, hidden_dim)

        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        u_idx: torch.Tensor,
        g_idx: torch.Tensor,
        tab_f: torch.Tensor,
        txt_f: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Menghitung forward pass probabilitas rekomendasi dan bobot atensi.

        Args:
            u_idx (torch.Tensor): Tensor indeks identitas pengguna.
            g_idx (torch.Tensor): Tensor indeks identitas game.
            tab_f (torch.Tensor): Matriks fitur tabular metadata game.
            txt_f (torch.Tensor): Matriks fitur embedding teks judul game.
            return_attention (bool, optional): Mengembalikan bobot atensi jika bernilai True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Skor probabilitas rekomendasi
            dan bobot atensi modalitas (jika diminta).
        """
        u = self.u_emb(u_idx)
        g = self.g_emb(g_idx)
        t = self.tab_proj(tab_f)
        x = self.txt_proj(txt_f)

        modalities = torch.stack([g, t, x], dim=1)

        query = self.q_layer(u).unsqueeze(1)
        keys = self.k_layer(modalities)

        att_scores = torch.bmm(query, keys.transpose(1, 2))
        att_weights = F.softmax(att_scores, dim=-1)

        context = torch.bmm(att_weights, modalities).squeeze(1)

        pred = self.pred_layer(context).squeeze(-1)

        if return_attention:
            return pred, att_weights.squeeze(1)
        return pred
