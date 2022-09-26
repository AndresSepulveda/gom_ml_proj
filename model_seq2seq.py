import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):    
        self.X = torch.tensor(data[0]).float()
        self.Y = torch.tensor(data[1]).float()  
        self.offset = torch.tensor(data[2]).float()
        self.XY = torch.tensor(data[3]).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        input_data = self.X[index,...]
        output_target = self.Y[index,...]
        offset = self.offset[index,...]
        sub_input = self.XY[index,...]
      
        return (input_data, output_target, offset, sub_input)
    

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_size, dropout=0, num_layers=1):
        '''
        seq_len: input sequence length
        n_features: number of features / time series
        hidden_size: size of hidden state
        num_layers: layers of recurrent layers
        '''
        super(Encoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=n_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=False,
            dropout = dropout
        )
        
        
    def forward(self, x):
        '''
        here 'batch_first = False', so the input dims = [seq_len, number of samples/batch, number of features]
        return hidden and cell dims = [number of layers, batch size, hidden size]
        '''
        
        hidden_0 = Variable(torch.zeros(self.num_layers, x.shape[1], self.hidden_size))
        x, hidden = self.gru(x, hidden_0)
        return hidden
    
    
class Decoder(nn.Module):
    def __init__(self, n_input, hidden_size, dropout=0, num_layers=1):
        '''
        seq_len: input sequence length
        n_features: number of features / time series
        hidden_size: size of hidden state
        num_layers: layers of recurrent layers
        '''
        
        super(Decoder, self).__init__()
        
        self.n_input = n_input
        self.gru = nn.GRU(
            input_size=n_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_size, n_input)
        

    def forward(self, x, input_hidden):
        '''
        x: number of previous steps, batch size, n_features
        input_hidden, input_cell from encoder output
        return x, number of predicted steps, batch size, n_features
        '''
        x, hidden_n = self.gru(x, input_hidden)
        x = self.output_layer(x) 
        return x, hidden_n
    

class Seq2Seq(nn.Module):
    def __init__(self, seq_len, label_len, n_features, n_output, hidden_size=32, num_layers=1, teacher_forcing=0, dropout=0):
        super(Seq2Seq, self).__init__()
        
        self.n_output, self.label_len = n_output, label_len
        self.encoder = Encoder(n_features, hidden_size, dropout, num_layers)
        self.decoder = Decoder(n_output, hidden_size, dropout, num_layers)
        self.teacher_forcing = teacher_forcing
        
        
    def forward(self, x, xy):  
        '''
        x: sequence length, batch size, n_features
        xy: label length, batch size, n_features
        '''
        hidden = self.encoder(x)        
        outputs = torch.zeros(self.label_len, x.shape[1], self.n_output)
        
        if self.teacher_forcing < 0:                 # 1. use true value
            for t in range(self.label_len):
                prev_x, hidden = self.decoder(xy[t:t+1,...], hidden)
                outputs[t,...] = prev_x
        elif self.teacher_forcing > 1:               # 2. recursive
            tmp_inp = xy[0:1,...]
            for t in range(self.label_len):
                prev_x, hidden = self.decoder(tmp_inp, hidden)
                outputs[t,...] = prev_x
                tmp_inp = prev_x
        else:                                        # 3. mixed method
            tmp_inp = xy[0:1,...]
            for t in range(self.label_len):
                if random.random() < self.teacher_forcing:
                    tmp_inp = xy[t:t+1,...]
                    prev_x, hidden = self.decoder(tmp_inp, hidden)   
                    tmp_inp = prev_x
                else:
                    prev_x, hidden = self.decoder(tmp_inp, hidden)
                    tmp_inp = prev_x
                outputs[t,...] = prev_x
                
        return outputs