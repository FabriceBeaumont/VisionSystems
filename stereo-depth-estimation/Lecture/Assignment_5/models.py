import torch
import torch.nn as nn
import torch.jit as jit

# The model classes are included in this file
# Our custom LSTM and GRU have the same structure:
# Lowest level: customLSTMCell() and customGRUCell():
# --> implements an individual cell
# Middle level: customLSTM() and customGRU():
# --> passes inputs through several layers of cells
# Highest level: customLSTMClassifier() and customGRUClassifier():
# --> applies encoding, RNN and then classification
# JIT is used in an attempt to accelerate the RNN loop!
# Finally there is also LSTMClassifier(), which uses the pyTorch LSTM


# LSTM
class customLSTMCell(jit.ScriptModule):
    def __init__(self, embed_dim, hidden_dim, bias=True):
        super().__init__()
        # Separate biases are probably okay here
        # Linear layers forget gate
        self.xf = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hf = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers input gate
        self.xi = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hi = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers update gate
        self.xC = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hC = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers output gate
        self.xo = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.ho = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    @jit.script_method
    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = state
        f = self.sigmoid(self.xf(x)+self.hf(hx))
        i = self.sigmoid(self.xi(x)+self.hi(hx))
        C_tmp = self.tanh(self.xC(x)+self.hC(hx))
        C = f*cx + i*C_tmp
        o = self.sigmoid(self.xo(x)+self.ho(hx))
        h = o * self.tanh(C)
        return h, C


class customLSTM(jit.ScriptModule):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, bias=True, batch_first=True):
        super().__init__()
        self.cells = [customLSTMCell(embed_dim, hidden_dim, bias=bias)] # Could also use nn.LSTMCell() here
        self.cells += [customLSTMCell(hidden_dim, hidden_dim, bias=bias) for i in range(num_layers-1)]
        self.cells = nn.ModuleList(self.cells)
        
    @jit.script_method
    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        b, s, p = x.shape
        hx, cx = state
        out = jit.annotate(List[Tensor], [])
        for i in range(s):
            x_in = x[:, i]
            for j, LSTMCell in enumerate(self.cells):
                x_in, cx[j] = LSTMCell(x_in, (hx[j].clone(), cx[j].clone()))
                hx[j] = x_in
            out.append(x_in)
        out = torch.stack(out, 1)
        return out, (hx, cx)


class customLSTMClassifier(jit.ScriptModule):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.Sequential(*[nn.Linear(input_dim, embed_dim), nn.PReLU()])
        self.rnn = customLSTM(embed_dim, self.hidden_dim, self.num_layers, bias=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    @jit.script_method
    def init_state(self, batch_size):
        # type: (int) -> Tuple[Tensor, Tensor]
        return (torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device), torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device))
    
    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor
        b, c, h, w = x.shape
        x = x.view(b, c*h, w)
        x = self.encoder(x)
        init_states = self.init_state(b)
        out, _ = self.rnn(x, init_states) 
        return self.classifier(out[:,-1])

    
# GRU    
class customGRUCell(jit.ScriptModule):
    def __init__(self, embed_dim, hidden_dim, bias=True):
        super().__init__()
        # Separate biases are probably okay here
        # Linear layers forget gate
        self.xz = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hz = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers input gate
        self.xr = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hr = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers update gate
        self.xh = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hh = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    @jit.script_method
    def forward(self, x, state):
        # type: (Tensor, Tensor) -> Tensor
        hx = state
        z = self.sigmoid(self.xz(x)+self.hz(hx))
        r = self.sigmoid(self.xr(x)+self.hr(hx))
        n = self.tanh(self.xh(x) + r*self.hh(hx))
        h = (1 - z)*n + z*hx
        return h


class customGRU(jit.ScriptModule):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, bias=True, batch_first=True):
        super().__init__()
        self.cells = [customGRUCell(embed_dim, hidden_dim, bias=bias)]
        self.cells += [customGRUCell(hidden_dim, hidden_dim, bias=bias) for i in range(num_layers-1)]
        self.cells = nn.ModuleList(self.cells)
        
    @jit.script_method
    def forward(self, x, state):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        b, s, p = x.shape
        hx = state
        out = jit.annotate(List[Tensor], [])
        for i in range(s):
            x_in = x[:, i]
            for j, GRUCell in enumerate(self.cells):
                x_in = GRUCell(x_in, hx[j].clone())
                hx[j] = x_in
            out.append(x_in)
        out = torch.stack(out, 1)
        return out, hx


class customGRUClassifier(jit.ScriptModule):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.Sequential(*[nn.Linear(input_dim, embed_dim), nn.PReLU()])
        self.rnn = customGRU(embed_dim, self.hidden_dim, self.num_layers, bias=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @jit.script_method
    def init_state(self, batch_size):
        # type: (int) -> Tensor
        return torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device)
    
    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor
        b, c, h, w = x.shape
        x = x.view(b, c*h, w)
        x = self.encoder(x)
        init_state = self.init_state(b)
        out, _ = self.rnn(x, init_state) 
        return self.classifier(out[:,-1])
    
    
# LSTM using nn.LSTM()
class LSTMClassifier(jit.ScriptModule):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.Sequential(*[nn.Linear(input_dim, embed_dim), nn.PReLU()])
        self.rnn = nn.LSTM(embed_dim, self.hidden_dim, self.num_layers, bias=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @jit.script_method
    def init_state(self, batch_size):
        # type: (int) -> Tuple[Tensor, Tensor]
        return (torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device), torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device))

    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor
        b, c, h, w = x.shape
        x = x.view(b, c*h, w)
        x = self.encoder(x)
        init_states = self.init_state(b)
        out, _ = self.rnn(x, init_states) 
        return self.classifier(out[:,-1])
    
# LSTM but without jit
class nojit_customLSTMCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True):
        super().__init__()
        # Separate biases are probably okay here
        # Linear layers forget gate
        self.xf = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hf = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers input gate
        self.xi = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hi = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers update gate
        self.xC = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.hC = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Linear layers output gate
        self.xo = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.ho = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, state):
        hx, cx = state
        f = self.sigmoid(self.xf(x)+self.hf(hx))
        i = self.sigmoid(self.xi(x)+self.hi(hx))
        C_tmp = self.tanh(self.xC(x)+self.hC(hx))
        C = f*cx + i*C_tmp
        o = self.sigmoid(self.xo(x)+self.ho(hx))
        h = o * self.tanh(C)
        return h, C


class nojit_customLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, bias=True, batch_first=True):
        super().__init__()
        self.cells = [nojit_customLSTMCell(embed_dim, hidden_dim, bias=bias)]
        self.cells += [nojit_customLSTMCell(hidden_dim, hidden_dim, bias=bias) for i in range(num_layers-1)]
        self.cells = nn.ModuleList(self.cells)
        
    def forward(self, x, state):
        b, s, p = x.shape
        hx, cx = state
        out = []
        for i in range(s):
            x_in = x[:, i]
            for j, LSTMCell in enumerate(self.cells):
                x_in, cx[j] = LSTMCell(x_in, (hx[j].clone(), cx[j].clone()))
                hx[j] = x_in
            out.append(x_in)
        out = torch.stack(out, 1)
        return out, (hx, cx)

    
class nojit_customLSTMClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.Sequential(*[nn.Linear(input_dim, embed_dim), nn.PReLU()])
        self.rnn = nojit_customLSTM(embed_dim, self.hidden_dim, self.num_layers, bias=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_state(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device), torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device))
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c*h, w)
        x = self.encoder(x)
        init_states = self.init_state(b)
        out, _ = self.rnn(x, init_states) 
        return self.classifier(out[:,-1])