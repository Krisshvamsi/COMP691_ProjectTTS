
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def dynamic_batch_collate(batch):
    """
    Collates batches dynamically based on the length of sequences within each batch.
    This function ensures that each batch contains sequences of similar lengths,
    optimizing padding and computational efficiency.

    Args:
        batch: A list of dictionaries, each containing 'id', 'phoneme_seq_encoded',
               'mel_spectrogram', 'mel_length', 'stop_token_targets'.

    Returns:
        A batch of sequences where sequences are padded to match the longest sequence in the batch.
    """
    # Sort the batch by 'mel_length' in descending order for efficient packing
    batch.sort(key=lambda x: x['mel_lengths'], reverse=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract sequences and their lengths
    ids = [item['id'] for item in batch]
    phoneme_seqs = [item['phoneme_seq_encoded'] for item in batch]
    mel_specs = [item['mel_spec'] for item in batch]
    #bos_mel_specs = [item['bos_mel_spectrogram'] for item in batch]
    #eos_mel_specs = [item['eos_mel_spectrogram'] for item in batch]
    mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], device=device)
    stop_token_targets = [item['stop_token_targets'] for item in batch]

    # Pad phoneme sequences
    phoneme_seq_padded = torch.nn.utils.rnn.pad_sequence(phoneme_seqs, batch_first=True, padding_value=0).to(device)

    # Find the maximum mel length for padding
    max_len = max(mel_lengths).item()
    num_mel_bins = 80  
    
    mel_specs_padded = torch.zeros((len(mel_specs), num_mel_bins, max_len), device=device)
    for i, mel in enumerate(mel_specs):
        mel_len = mel.shape[1]
        mel_specs_padded[i, :, :mel_len] = mel.to(device)
    
    # # Pad mel spectrograms
    # bos_mel_specs_padded = torch.zeros((len(bos_mel_specs), num_mel_bins, max_len), device=device)
    # for i, mel in enumerate(bos_mel_specs):
    #     mel_len = mel.shape[1]
    #     bos_mel_specs_padded[i, :, :mel_len] = mel.to(device)
    #     
    # eos_mel_specs_padded = torch.zeros((len(eos_mel_specs), num_mel_bins, max_len), device=device)
    # for i, mel in enumerate(eos_mel_specs):
    #     mel_len = mel.shape[1]
    #     eos_mel_specs_padded[i, :, :mel_len] = mel.to(device)

    # Pad stop token targets
    stop_token_targets_padded = torch.zeros((len(stop_token_targets), max_len), device=device)
    for i, stop in enumerate(stop_token_targets):
        stop_len = stop.size(0)
        stop_token_targets_padded[i, :stop_len] = stop.to(device)

    return ids, phoneme_seq_padded, mel_specs_padded, mel_lengths, stop_token_targets_padded


class EncoderPrenet(torch.nn.Module):
     """
    Module for the encoder prenet in the Transformer-based TTS system.

    This module consists of several convolutional layers followed by batch normalization,
    ReLU activation, and dropout. It then performs a linear projection to the desired dimension.

    Parameters:
        input_dim (int): Dimension of the input features. Defaults to 512.
        hidden_dim (int): Dimension of the hidden layers. Defaults to 512.
        num_layers (int): Number of convolutional layers. Defaults to 3.
        dropout (float): Dropout probability. Defaults to 0.2.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim). """
     def __init__(self, input_dim=512, hidden_dim=512, num_layers=3, dropout=0.2):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm1d(hidden_dim))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
        self.conv_layers = nn.Sequential(*conv_layers)

        # Final linear projection
        self.projection = nn.Linear(hidden_dim, hidden_dim)

     def forward(self, x):
        x = x.transpose(1, 2)  # Transpose for convolutional layers (Batch, SeqLen, Channels)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # Transpose back
        x = self.projection(x)
        return x
    
    
class DecoderPrenet(torch.nn.Module):
    
    """
    Module for the decoder prenet in the Transformer-based TTS system.

    This module consists of two fully connected layers followed by ReLU activation,
    and performs a linear projection to the desired output dimension.

    Parameters:
        input_dim (int): Dimension of the input features. Defaults to 80.
        hidden_dim (int): Dimension of the hidden layers. Defaults to 256.
        output_dim (int): Dimension of the output features. Defaults to 512.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim). """
    
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.projection(x)

        return x
    

class ScaledPositionalEncoding(nn.Module):
    
    """
    Module for adding scaled positional encoding to input sequences.

    Parameters:
        d_model (int): Dimensionality of the model. It must match the embedding dimension of the input.
        max_len (int): Maximum length of the input sequence. Defaults to 5000.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

    Returns:
        torch.Tensor: Output tensor with scaled positional encoding added, shape (batch_size, seq_len, embedding_dim). """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Adds scaled positional encoding to input tensor x.
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        scaled_pe = self.pe[:x.size(0), :, :] * self.scale
        x = x + scaled_pe
        return x

    
class PostNet(torch.nn.Module):
    
    """
    Post-processing network for mel-spectrogram enhancement.

    This module consists of multiple convolutional layers with batch normalization and ReLU activation.
    It is used to refine the mel-spectrogram output from the decoder.

    Parameters:
        mel_channels (int): Number of mel channels in the input mel-spectrogram.
        postnet_channels (int): Number of channels in the postnet layers.
        kernel_size (int): Size of the convolutional kernel.
        postnet_layers (int): Number of postnet layers.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, mel_channels).

    Returns:
        torch.Tensor: Output tensor with refined mel-spectrogram, shape (batch_size, seq_len, mel_channels). """

    
    def __init__(self, mel_channels, postnet_channels, kernel_size, postnet_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(mel_channels, postnet_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(postnet_channels),
                nn.ReLU()
            )
        )
        
        # Middle layers
        for _ in range(1, postnet_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(postnet_channels, postnet_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(postnet_channels),
                    nn.ReLU()
                )
            )
        
        # Final layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(postnet_channels, mel_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(mel_channels)
            )
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        return x
