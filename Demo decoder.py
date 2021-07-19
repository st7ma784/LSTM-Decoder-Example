class LSTMAttnDecoderRNN(nn.Module):
    def __init__(self,embedding, hidden_size, output_size):
        self.embedding =embedding ##nn.Embedding(output_size, hidden_size) #loaded weights directly from from CLIP
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size * 2, hidden_size)
        self.LSTM=nn.LSTM(input_size=512, hidden_size=512, num_layers=layers)
        self.attn3 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer1=nn.Linear(hidden_size, hidden_size)
        self.layer2=nn.Linear(hidden_size, hidden_size)
        self.layer3=nn.Linear(hidden_size,2*hidden_size)
        self.layer4=nn.Linear(2*hidden_size,hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
  def forward(self, input, hidden, encoder_outputs):
      embedded = self.embedding(input)
      att1=self.layer1(embedded)
      attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
      attn_applied = attn_weights*encoder_outputs
      output = torch.cat((embedded, attn_applied))  
      output = self.attn_combine(output)
      output = F.relu(output)
      output=self.layer2(output)
      output= F.softmax(self.attn3(torch.cat((output,att1)),dim=1)
      output, hidden = self.gru(output,hidden)
      output=self.out(self.layer4(self.layer3(output)))
      output =self.softmax(output[0])
      hidden=hidden.squeeze()
      return output, hidden, attn_weights
