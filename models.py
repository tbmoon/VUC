import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    '''
    Produce N identical layers.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'.
    '''
    # query, key, value: [batch_size, n_heads, frame_length, d_k]
    # scores: [batch_size, n_heads, frame_length, frame_length]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # pattn: [batch_size, n_heads, frame_length, frame_length]
    # value: [batch_size, n_heads, frame_length, d_k]
    # weighted sum: [batch_size, n_heads, frame_length, d_k]
    weighted_sum = torch.matmul(p_attn, value)
    return weighted_sum, p_attn


class Embedding(nn.Module):
    '''
    Implement the embedding function of video feature.
    '''
    def __init__(self, rgb_feature_size, audio_feature_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.fc_embed = nn.Linear(rgb_feature_size + audio_feature_size, d_model)

    def forward(self, frame_features):
        '''
        inputs:
            - frame_features: [batch_size, frame_length, feature_size]
        outputs:
            - embedded: [batch_size, frame_length, d_model]
        '''
        embedded = self.fc_embed(frame_features)
        '''
        In the embedding layers, we multiply those weights
        by the square root of 'd_model'.
        '''
        return embedded * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    '''
    Implement the positional encoding function.
    '''
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        '''
        Take in number of heads and model size.
        Assume d_v always equals d_k.
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        '''
        Implement Figure 2.
        '''
        batch_size = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => n_heads x d_k
        # query, key, value: [batch_size, frame_length, d_model]
        # query, key, value: [batch_size, n_heads, frame_length, d_k]
        query, key, value = \
            [l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        
        # 3) 'Concat' using a view and apply a final linear.
        # x: [batch_size, n_heads, frame_length, d_k]
        # x: [batch_size, frame_length, d_model]
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    '''
    Implement FFN equation.
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    '''
    Construct a layernorm module.
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    '''
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        Apply residual connection to any sublayer with the same size.
        '''
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    '''
    Encoder is made up of self-attn and feed forward.
    '''
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x):
        '''
        Follow Figure 1 (left) for connections.
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    '''
    Core encoder is a stack of N layers.
    '''
    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x):
        '''
        Pass the input through each layer in turn.
        '''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Frame2Segment(nn.Module):
    '''
    Translate frame into segment level.
    '''
    def __init__(self):
        super(Frame2Segment, self).__init__()
        self.avgpool1d = nn.AvgPool1d(kernel_size=5, stride=5, padding=0)
        self.maxpool1d = nn.MaxPool1d(kernel_size=5, stride=5, padding=0)

    def forward(self, frame_features):
        '''
        inputs:
            - frame_features: [batch_size, frame_length, d_model]
        outputs:
            - seg_features: [batch_size, seg_length, d_model]
        '''
        frame_features = frame_features.transpose(1, 2)  # frame_features: [batch_size, d_model, frame_length]
        seg_features = self.avgpool1d(frame_features)    # seg_features: [batch_size, d_model, seg_length=60]
        seg_features = seg_features.transpose(1, 2)      # seg_features: [batch_size, seg_length=60, d_model]        
        return seg_features


def seg_attention(query, key, value, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'.
    '''
    # query, key, value: [batch_size, frame_length, d_proj]
    # scores: [batch_size, frame_length]
    # p_attn: [batch_size, frame_length]
    d_proj = query.size(-1)
    scores = torch.sum(query * key, dim=-1) / math.sqrt(d_proj)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # weighted_sum: [batch_size, frame_length, d_proj]
    # p_attn: [batch_size, frame_length]
    weighted_sum = p_attn.unsqueeze(2) * value
    return weighted_sum, p_attn


class SegmentAttention(nn.Module):
    def __init__(self, d_model, d_proj, dropout=0.1):
        '''
        Take in number of projection size.
        '''
        super(SegmentAttention, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_proj), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        '''
        Implement Figure 2.
        '''
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => d_proj
        # query, key: [batch_size, seg_length, d_model]
        # query, key, value: [batch_size, seg_length, d_proj]
        query, key = \
            [l(x) for l, x in zip(self.linears, (query, key))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = seg_attention(query, key, value, dropout=self.dropout)
        
        # 3) Apply element-wise summation w.r.t. seg_length.
        x = torch.sum(x, dim=1, keepdim=True)  # x: [batch_size, 1, d_proj]
        return x


class Classifier(nn.Module):
    '''
    Classify video labels.
    '''
    def __init__(self, d_model, d_proj, n_attns, num_classes, dropout=0.1):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(d_model, d_proj)
        self.seg_attns = clones(SegmentAttention(d_model, d_proj, dropout), n_attns)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_classes, kernel_size=d_proj)
        self.maxpool1d = nn.MaxPool1d(kernel_size=n_attns, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg_features, device):
        '''
        Calculated by weighted sum using hadamard product.
        inputs:
            - seg_features: [batch_size, seg_length, d_model]
        outputs:
            - vid_probs: [batch_size, num_classes]
        '''
        vid_logits = torch.Tensor().to(device)
        seg_values = self.linear(seg_features)                              # seg_values: [batch_size, seg_length, d_proj]
        for seg_attn in self.seg_attns:
            vid_feature = seg_attn(seg_features, seg_features, seg_values)  # vid_feature: [batch_size, 1, d_proj]
            vid_logit = self.conv1d(self.dropout(F.relu(vid_feature)))      # vid_logit: [batch_size, num_classes, 1]
            vid_logits = torch.cat((vid_logits, vid_logit), dim=2)          # vid_logits: [batch_size, num_classes, n_attns]
        vid_logits = self.maxpool1d(vid_logits).squeeze(2)                  # vid_logits: [batch_size, num_classes]
        vid_probs = self.sigmoid(vid_logits)                                # vid_probs: [batch_size, num_classes]
        return vid_probs


class TransformerModel(nn.Module):
    '''
    Implement model based on the transformer.
    '''
    def __init__(self, n_layers, n_heads, rgb_feature_size,
                 audio_feature_size, d_model, d_ff, d_proj, n_attns,
                 num_classes, dropout):
        super(TransformerModel, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        self.embedding = Embedding(rgb_feature_size, audio_feature_size, d_model)
        self.position = PositionalEncoding(d_model, dropout)
        self.attn = MultiHeadedAttention(n_heads, d_model)
        self.pff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, c(self.attn), c(self.pff), dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.frame2seg = Frame2Segment()
        self.classifier = Classifier(d_model, d_proj, n_attns, num_classes, dropout)

    def forward(self, padded_frame_rgbs, padded_frame_audios, device):
        '''
        inputs:
            - padded_frame_rgbs: [batch_size, frame_length, rgb_feature_size]
            - padded_frame_audios: [batch_size, frame_length, audio_feature_size]
        outputs:
            - vid_probs: [batch_size, num_classes]
        '''
        padded_frame_rgbs = F.normalize(padded_frame_rgbs, p=2, dim=2)
        padded_frame_audios = F.normalize(padded_frame_audios, p=2, dim=2)

        # frame_features: [batch_size, frame_length, rgb_feature_size + audio_feature_size]
        frame_features = torch.cat((padded_frame_rgbs, padded_frame_audios), 2)
        frame_features = self.embedding(frame_features)     # frame_features: [batch_size, frame_length, d_model]
        frame_features = self.position(frame_features)
        frame_features = self.encoder(frame_features)
        seg_features = self.frame2seg(frame_features)       # seg_features: [batch_size, seg_length=60, d_model]
        vid_probs = self.classifier(seg_features, device)   # vid_probs: [batch_size, num_classes]
        return vid_probs
