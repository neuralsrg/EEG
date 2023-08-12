import sys
sys.path.append('/kaggle/working/PyTorchWavelets/')

import math
from typing import Optional, List

import torch
import torchaudio
import torch.nn as nn

from wavelets_pytorch.transform import WaveletTransformTorch
from wavelets_pytorch.wavelets import Morlet

from sklearn.decomposition import PCA


class ConvolutionModule(torch.nn.Module):
    d_model: int
    kernel_size: int
    conv_module_dropout: float

    def __init__(self, d_model: int, kernel_size: int, conv_module_dropout: float):
        '''
        :param int d_model: Input dimension
        :param int kernel_size: Kernel size of Depthwise Convolution
        :param float dropout: Dropout probability 
        '''
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pointwise_conv_1 = nn.Conv1d(self.d_model, 2 * self.d_model, kernel_size=1)
        self.activation_1 = nn.GLU()
        self.depthwise_conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=self.kernel_size, groups=self.d_model, padding='same')
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.activation_2 = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.dropout = nn.Dropout(conv_module_dropout)
        
        self.reset_parameters()

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        :param torch.Tensor x: (batch, time, d_model)
        :param torch.Tensor pad_mask: (batch, time) takes True value for the positions corresponding to the padding
        :return: (batch, time, d_model)
        :rtype: torch.Tensor
        '''
        
        x = self.layer_norm(x)
        x = self.pointwise_conv_1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation_1(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)

        x = self.depthwise_conv(x.permute(0, 2, 1))
        x = self.batch_norm(x)
        x = self.activation_2(x)
        x = self.pointwise_conv_2(x).permute(0, 2, 1)
        x = self.dropout(x)

        return x
    
    def reset_parameters(self):
        pw_max = self.d_model ** -0.5
        dw_max = self.kernel_size ** -0.5
        with torch.no_grad():
            torch.nn.init.uniform_(self.pointwise_conv_1.weight, -pw_max, pw_max)
            torch.nn.init.uniform_(self.pointwise_conv_2.weight, -pw_max, pw_max)
            torch.nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            
            torch.nn.init.uniform_(self.pointwise_conv_1.bias, -pw_max, pw_max)
            torch.nn.init.uniform_(self.pointwise_conv_2.bias, -pw_max, pw_max)
            torch.nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)
            
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, emb_dropout: float, in_seq_len: int):
        super().__init__()
        
        den = torch.exp(- torch.arange(0, d_model, 2)* math.log(10000) / d_model)
        pos = torch.arange(0, in_seq_len).reshape(in_seq_len, 1)
        pos_embedding = torch.zeros((in_seq_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, in_seq_len, d_model)

        self.dropout = nn.Dropout(emb_dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: torch.Tensor):
        return self.dropout(x + self.pos_embedding)


class E2STransformer(nn.Module):
    
    def __init__(self, n_channels: int, n_wvt_bins: int, d_model: int,
                 kernel_size: int, conv_module_dropout: int, emb_dropout: float,
                 in_seq_len: int, n_fft: int, hop_size: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float,
                 activation: str, audio_sr: int, audio_paths: List[str],
                 eeg_sr: int, dj: float, example_input: torch.tensor):
        """
        :param dict config: dictionart with all model parameters
        :param List[str] audio_paths: list of audio file paths to fit PCA on
        :param torch.tensor example_input: input to compute wavelet filters on. Should have shape (n_channels, in_seq_len)
        """
        super().__init__()

        self.conv_downsampling = torch.nn.Conv1d(n_channels, 1, kernel_size=1) # (N, c_in, L) -> (N, 1, L)
        self.ln = nn.LayerNorm(n_wvt_bins)
        self.ffn = nn.Linear(n_wvt_bins, d_model)
        self.conv_module = ConvolutionModule(d_model=d_model, kernel_size=kernel_size,
                                             conv_module_dropout=conv_module_dropout)
        self.positional_encoding = PositionalEncoding(d_model=d_model, emb_dropout=emb_dropout,
                                                      in_seq_len=in_seq_len)
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.d_model = d_model
        self.eeg_sr = eeg_sr
        self.dj = dj
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.audio_sr = audio_sr
        self.compute_pca_components(audio_paths)
        
        # Specials
        self.src_sos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        self.src_eos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        self.tgt_sos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        self.tgt_eos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        
        self.reset_parameters()
        self.get_wavelet_filters(example_input)
        
    def reset_parameters(self):
        pw_max = self.d_model ** -0.5
        with torch.no_grad():
            torch.nn.init.uniform_(self.src_sos, -pw_max, pw_max)
            torch.nn.init.uniform_(self.src_eos, -pw_max, pw_max)
            torch.nn.init.uniform_(self.tgt_sos, -pw_max, pw_max)
            torch.nn.init.uniform_(self.tgt_eos, -pw_max, pw_max)
    
    def get_wavelet_filters(self, x):
        """
        Computes Wavelet convolution weights
        :param torch.tensor x: example input of shape (n_channels, in_seq_len)
        """
        wvt_transformer = WaveletTransformTorch(
            dt=self.eeg_sr,
            dj=self.dj,
            wavelet=Morlet(),
            cuda=torch.cuda.is_available()
        )
        _ = wvt_transformer.cwt(x)
        self.filters = nn.ModuleList(wvt_transformer._extractor._filters)  # requires_grad: False
        # self.register_buffer('filters', filters)
        
    def compute_pca_components(self, audio_paths):
        """
        :param List[str] audio_paths: list of audio file paths to fit PCA on
        """
        audios_srs = [torchaudio.load(path) for path in audio_paths]
        all_audios = []
        for audio, sr in audios_srs:
            if sr != self.audio_sr:
                audio = torchaudio.functional.resample(waveform=audio, orig_freq=sr, new_freq=self.audio_sr)[0]
            all_audios.append(audio)
        
        all_audios = torch.cat(all_audios)
        all_audios = torch.stft(all_audios, n_fft=self.n_fft, hop_length=self.hop_size, return_complex=True)  # (n_freq_bins, n_frames)
        all_audios = torch.abs(all_audios).t().numpy()
        
        pca = PCA(n_components=self.d_model)
        pca.fit(all_audios)
        
        components = torch.tensor(pca.components_)  # (d_model, n_freq_bins)
        mean = torch.tensor(pca.mean_)  # (n_freq_bins)
        
        self.register_buffer('components', components)
        self.register_buffer('mean', mean)
        
    def cwt(self, x):
        """
        Computes continuous wavelet transform of a given tensor
        :param torch.tensor x: input of shape (batch, n_channels, in_seq_len)
        :return torch.tensor out: cwt result of shape (batch, n_channels, n_wvt_bins, in_seq_len)
        """

        batch, n_channels, signal_length = x.size()
        x = x.view(batch * n_channels, signal_length).unsqueeze(1)  # (N, 1, in_seq_len)

        # x = x.type(torch.FloatTensor)
        # x.requires_grad_(requires_grad=False)

        results = [None]*len(self.filters)
        for ind, conv in enumerate(self.filters):
            results[ind] = conv(x)
            
        results = torch.stack(results)     # [n_scales,n_batch,2,t]
        results = results.permute(1,0,2,3) # [n_batch,n_scales,2,t]

        # results = torch.abs(results[:,:,0,:] + results[:,:,1,:]*1j
        results = (results[:,:,0,:]**2 + results[:,:,1,:]**2)**0.5
        
        results = results.reshape(batch, n_channels, results.size(1), signal_length)
        return results
        
    def prepare_src(self, x):
        """
        :param torch.tensor x: input of shape (batch_size, n_channels, in_seq_len)
        :rtype torch.tensor
        :return out of shape (batch_size, in_seq_len, d_model)
        """
        
        # Wavelet Transform
        out = self.cwt(x)  # (batch_size, n_channels, n_wvt_bins, in_seq_len)
        batch_size, n_channels, n_wvt_bins, in_seq_len = out.size()
        
        # Convolution downsampling
        out = out.permute(0, 3, 1, 2)  # (batch_size, in_seq_len, n_channels, n_wvt_bins)
        out = out.reshape(batch_size * in_seq_len, n_channels, n_wvt_bins)  # (batch_size * in_seq_len, n_channels, n_wvt_bins)
        out = self.conv_downsampling(out).squeeze(1)  # (batch_size * in_seq_len, n_wvt_bins)
        out = out.reshape(batch_size, in_seq_len, n_wvt_bins)  # (batch_size, in_seq_len, n_wvt_bins)
        
        # LayerNorm & Feed Forward
        out = self.ln(out)  # (batch_size, in_seq_len, n_wvt_bins)
        out = self.ffn(out)  # (batch_size, in_seq_len, d_model)
        
        # Convolution module from https://arxiv.org/pdf/2005.08100.pdf
        out = self.conv_module(out)  # (batch_size, in_seq_len, d_model)
        
        # Positional Encoding
        out = self.positional_encoding(out)  # (batch_size, in_seq_len, d_model)
        
        return out
    
    def prepare_tgt(self, x):  # Add some audio normalization???
        """
        :param torch.tensor x: input of shape (batch_size, audio_len)
        :rtype torch.tensor
        :return out of shape (batch_size, out_seq_len, d_model)
        """
        # n_freq_bins = self.n_fft // 2 + 1
        # out_seq_len = self.n_fft // self.hop_size + 1
        
        # STFT
        out = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, return_complex=True)  # (batch_size, n_freq_bins, out_seq_len)
        out = torch.abs(out.permute(0, 2, 1))  # (batch_size, out_seq_len, n_freq_bins)
        
        # PCA
        out = out - self.mean
        out = out @ self.components.t()  # (batch_size, out_seq_len, d_model)
        return out
        
    def forward(self, eeg, audio):
        """
        :param torch.tensor eeg: input of shape (batch_size, n_channels, in_seq_len)
        :rtype torch.tensor
        :return out of shape (batch_size, out_seq_len, n_freq_bins)
        """
        batch_size = eeg.size(0)
        src = self.prepare_src(eeg)  # (batch_size, in_seq_len, d_model)
        tgt = self.prepare_tgt(audio)  # (batch_size, out_seq_len, d_model)
        
        # Add <sos> and <eos>
        src = torch.cat((self.src_sos.repeat(batch_size, 1, 1), src, self.src_eos.repeat(batch_size, 1, 1)),
                        dim=1)  # (batch_size, 1 + in_seq_len + 1, d_model)
        tgt = torch.cat((self.tgt_sos.repeat(batch_size, 1, 1), tgt, self.tgt_eos.repeat(batch_size, 1, 1)),
                        dim=1)  # (batch_size, 1 + out_seq_len + 1, d_model)
        
        # Transformer
        causal_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(eeg.device).type(torch.bool)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=causal_mask)  # (batch_size, 1 + out_seq_len + 1, d_model)
        # out = out[:, 1:-1, :]  # (batch_size, out_seq_len, d_model)
        
        # Inverse PCA
        # out = out @ self.components  # (batch_size, out_seq_len, n_freq_bins)
        # out = out + self.mean
        
        return out, tgt