import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.weight_norm import weight_norm

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn( 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros( 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, d  = inputs.shape

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(n_s, -1)
        sigma = self.slots_logsigma.exp().expand(n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q.unsqueeze(0), k.unsqueeze(0)) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)


            updates = torch.einsum('bjd,bij->bid', v.unsqueeze(0), attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        print('slots', slots.size())

        return slots


class Attention(nn.Module):
    '''
    Calculates attention for each region of the image based on the query
    Reused code from original source https://github.com/hengyuan-hu/bottom-up-attention-vqa
    '''
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()
        self.nonlinear = nn.Linear(v_dim + q_dim, num_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)

        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits