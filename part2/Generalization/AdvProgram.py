##define adversarial reporogramming model

#define weight additive weight

class Program(nn.Module):
    def __init__(self, out_size):
        super(Program, self).__init__()
        
        self.weight = torch.nn.Parameter(data=torch.Tensor(3, *out_size))
        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = self.weight.mul(x)
        return x

#define module for adding weight to inputs

class AdvProgram(nn.Module):
    def __init__(self,in_size, out_size, mask_size, device=torch.device('cuda')):
        super(AdvProgram, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.program = Program(out_size).to(device)
        self.mask = torch.ones(3, *out_size, device=device)
        
    def forward(self, x):
        
        x = x + torch.tanh(self.program(self.mask))
        x = self.normalize_batch(x)
        
        return x
    
    def normalize_batch(self, input, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        
        ms = torch.tensor(means, device=str(input.device)).view(1,3,1,1)
        ss = torch.tensor(stds, device=str(input.device)).view(1,3,1,1)
        
        return (input-ms)/ss