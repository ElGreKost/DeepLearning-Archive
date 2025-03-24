import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x, init_states=None):
        bs, seq_len, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return self.dropout(hidden_seq), (h_t, c_t)

    def __repr__(self):
        return f"CustomLSTM(W_shape={self.W.shape}, U_shape={self.U.shape}, bias_shape={self.bias.shape}, dropout={self.dropout.p})"


class MultiLayerCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(MultiLayerCustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use ModuleDict to store each layer of CustomLSTM
        self.layers = nn.ModuleDict({
            f'lstm_layer_{i}': CustomLSTM(input_size if i == 0 else hidden_size, hidden_size,
                                          dropout=dropout if i < num_layers - 1 else 0.0)
            for i in range(num_layers)
        })

    def forward(self, x, init_states=None):
        layer_states = []
        if init_states is None:
            init_states = [(torch.zeros(x.size(0), self.hidden_size).to(x.device),
                            torch.zeros(x.size(0), self.hidden_size).to(x.device))
                           for _ in range(self.num_layers)]

        # Pass through each layer in sequence
        for i in range(self.num_layers):
            layer = self.layers[f'lstm_layer_{i}']
            x, (h_t, c_t) = layer(x, init_states[i])
            layer_states.append((h_t, c_t))  # Store hidden and cell states

        # Return the output from the final layer and states for all layers
        return x, layer_states

    def __repr__(self):
        layer_descriptions = "\n".join([f"{name}: {layer}" for name, layer in self.layers.items()])
        return f"MultiLayerCustomLSTM(\n  Layers:\n{layer_descriptions}\n)"


# Example usage
input_size = 10
hidden_size = 32
num_layers = 3
dropout = 0.3
batch_size = 4
seq_len = 5

# Instantiate the multi-layer custom LSTM with dropout
multi_layer_lstm = MultiLayerCustomLSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)

# Print the model to view all layers and weights
print(multi_layer_lstm)

# Create a dummy input tensor
dummy_input = torch.randn(batch_size, seq_len, input_size)

# Forward pass
hidden_seq, layer_states = multi_layer_lstm(dummy_input)
