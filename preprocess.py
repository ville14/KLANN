import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# create dataset
def PreProcess(train_input, train_target, sequence_length, truncate_length, batch_size):
    data = AudioDataSet(train_input, train_target, sequence_length, truncate_length)
    return DataLoader(dataset = data, batch_size = batch_size, shuffle = True)


class AudioDataSet(Dataset):
    def __init__(self, input, target, sequence_length, truncate_length):
        self.input_sequence = self.wrap_to_sequences(input, sequence_length, truncate_length)
        self.target_sequence = self.wrap_to_sequences(target, sequence_length, truncate_length)
        self.length = self.input_sequence.shape[0]

    def __getitem__(self, index):
        return self.input_sequence[index,:,:], self.target_sequence[index,:,:]

    def __len__(self):
        return self.length

    def wrap_to_sequences(self, waveform, sequence_length, truncate_length):
        num_sequences = int(np.floor((waveform.shape[1] - truncate_length) / sequence_length))
        tensors = []
        for i in range(num_sequences):
            low = i * sequence_length
            high = low + sequence_length + truncate_length
            tensors.append(waveform[0,low:high])
        return torch.unsqueeze(torch.stack(tensors, dim = 0), dim = -1)
