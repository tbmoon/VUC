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
    d_k = query.size(-1)
    # query, key, value: [batch_size, n_heads, frame_length, d_k]
    # scores: [batch_size, n_heads, frame_length, frame_length]
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


class TransformerEncoder(nn.Module):
    '''
    Implement model based on the transformer.
    '''
    def __init__(self, n_layers, n_heads, rgb_feature_size, audio_feature_size, d_rgb, d_audio, d_model, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        #self.rgb_dense = nn.Linear(rgb_feature_size, d_rgb)
        #self.audio_dense = nn.Linear(audio_feature_size, d_audio)
        #self.rgb_dense_bn = nn.BatchNorm1d(d_rgb)
        #self.audio_dense_bn = nn.BatchNorm1d(d_audio)
        #self.dropout = nn.Dropout(dropout)
        self.embedding = Embedding(rgb_feature_size, audio_feature_size, d_model)
        self.position = PositionalEncoding(d_model, dropout)
        self.attn = MultiHeadedAttention(n_heads, d_model)
        self.pff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, c(self.attn), c(self.pff), dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.avgpool1d = nn.AvgPool1d(kernel_size=5, stride=5, padding=0)
        self.maxpool1d = nn.MaxPool1d(kernel_size=5, stride=5, padding=0)

    def forward(self, padded_frame_rgbs, padded_frame_audios):
        '''
        inputs:
            - padded_frame_rgbs: [batch_size, frame_length, rgb_feature_size]
            - padded_frame_audios: [batch_size, frame_length, audio_feature_size]
        outputs:
            - seq_features: [batch_size, seq_length, d_model]
        '''
        #padded_frame_rgbs = self.rgb_dense(padded_frame_rgbs).transpose(1, 2)
        #padded_frame_rgbs = self.dropout(F.relu(self.rgb_dense_bn(padded_frame_rgbs).transpose(1, 2)))

        #padded_frame_audios = self.audio_dense(padded_frame_audios).transpose(1, 2)
        #padded_frame_audios = self.dropout(F.relu(self.audio_dense_bn(padded_frame_audios).transpose(1, 2)))

        padded_frame_rgbs = F.normalize(padded_frame_rgbs, p=2, dim=2)
        padded_frame_audios = F.normalize(padded_frame_audios, p=2, dim=2)

        # frame_features: [batch_size, frame_length, d_rgb + d_audio]
        frame_features = torch.cat((padded_frame_rgbs, padded_frame_audios), 2)

        frame_features = self.embedding(frame_features)  # frame_features: [batch_size, frame_length, d_model]
        frame_features = self.position(frame_features)
        frame_features = self.encoder(frame_features)

        frame_features = frame_features.transpose(1, 2)  # frame_features: [batch_size, d_model, frame_length]
        seq_features = self.avgpool1d(frame_features)    # seq_features: [batch_size, d_model, seq_length=60]
        seq_features = seq_features.transpose(1, 2)      # seq_features: [batch_size, seq_length=60, d_model]

        return seq_features


class Attn(nn.Module):
    '''
    Implement attention weights based on dot product. 
    '''
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def forward(self, decoder_output, encoder_outputs):
        '''        
        inputs:
            - decoder_output: [batch_size, 1, d_model]
            - encoder_outputs: [batch_size, seq_length=60, d_model]
        outputs:
            - raw_attn_weights: [batch_size, seq_length]
            - norm_attn_weights: [batch_size, 1, seq_length]
        '''
        # attn_energies: [batch_size, seq_length]
        #attn_energies = self.dot_score(decoder_output, encoder_outputs) 
        attn_energies = self.general_score(decoder_output, encoder_outputs)

        raw_attn_weights = self.sigmoid(attn_energies)                   # raw_attn_weights: [batch_size, seq_length]

        # 1) normalized by attention size.
        #attn_sum = torch.sum(raw_attn_weights, dim=1, keepdim=True) + 1e-6        
        #norm_attn_weights = raw_attn_weights / attn_sum                  # norm_attn_weights: [batch_size, seq_length]

        # 2) normalized by softmax function.
        norm_attn_weights = F.softmax(attn_energies, dim=1)              # attn_weights: [batch_size, seq_length]

        norm_attn_weights = norm_attn_weights.unsqueeze(1)               # norm_attn_weights: [batch_size, 1, seq_length]

        return raw_attn_weights, norm_attn_weights


class RNNDecoder(nn.Module):
    '''
    Implement decoder based on the RNN family.
    '''
    def __init__(self, d_model, d_linear, num_classes, dropout=0.1):
        super(RNNDecoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model)
        self.attn = Attn(d_model)
        self.context_bn = nn.BatchNorm1d(d_model)
        self.fc_layer1 = nn.Linear(2 * d_model, d_linear)
        self.fc_layer2 = nn.Linear(d_linear, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        '''
        - We run this one step (class) at a time.
        - The "contex" and "decoder_output" will be inputed to the next time step of the decoder.
        
        inputs:
            - decoder_intput: [1, batch_size, d_model]
            - decoder_hidden: [1, batch_size, d_model]
            - encoder_outputs: [batch_size, seq_lentgh=60, d_model]
        outputs:
            - raw_attn_weights: [batch_size, seq_length]
            - context: [1, batch_size, d_model]
            - decoder_output: [1, batch_size, d_model]
            - fc_layer2_output: [batch_size, num_classes]
        '''
        decoder_input = self.dropout(decoder_input)
        decoder_output, _ = self.gru(decoder_input, decoder_hidden)  # decoder_output: [1, batch_size, d_model]
        decoder_output = decoder_output.transpose(0, 1)              # decoder_output: [batch_size, 1, d_model]

        # raw_attn_weights: [batch_size, seq_length]
        # norm_attn_weigths: [batch_size, 1, seq_length]
        # context: [batch_size, 1, d_model]
        raw_attn_weights, norm_attn_weights = self.attn(decoder_output, encoder_outputs)
        context = norm_attn_weights.bmm(encoder_outputs) 

        decoder_output = decoder_output.squeeze(1)                   # decoder_output: [batch_size, d_model]
        context = context.squeeze(1)                                 # context: [batch_size, d_model]
        context = self.context_bn(context)

        fc_layer1_input = torch.cat((decoder_output, context), 1)    # fc_layer1_input: [batch_size, 2 * d_model]
        fc_layer1_output = F.relu(self.fc_layer1(fc_layer1_input))   # fc_layer1_output: [batch_size, d_linear]

        fc_layer2_output = self.fc_layer2(fc_layer1_output)          # fc_layer2_output: [batch_size, num_classes]        
        #fc_layer2_output = F.softmax(fc_layer2_output, dim=1)

        context = context.unsqueeze(0)                               # context: [1, batch_size, d_model]
        decoder_output = decoder_output.unsqueeze(0)                 # decoder_output: [1, batch_size, d_model]

        return raw_attn_weights, context, decoder_output, fc_layer2_output

    def init_input_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.d_model).to(device), torch.zeros(1, batch_size, self.d_model).to(device)
