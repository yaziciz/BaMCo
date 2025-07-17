from torch import nn
from .spatial_pooling_projector import SpatialPoolingProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x
    
class KGProjector(nn.Module):
    def __init__(self, config):
        super(KGProjector, self).__init__()
        self.linear = nn.Linear(512, config.hidden_size//2)
        self.linear_2 = nn.Linear(config.hidden_size//2, config.hidden_size)
        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        #self.kg_dim = config.kg_dim
        #self.kg_poll_size = config.kg_pool_size
    def forward(self, x):
        #channel first
        x = self.linear(x)
        x = self.relu(x)
        x = self.d1(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.d2(x)
        return x
    @property
    def proj_out_num(self):
        num = 1 #self.kg_dim // self.kg_poll_size
        return num
    
class ImageProjector(nn.Module):
    def __init__(self, config):
        super(ImageProjector, self).__init__()
        self.linear = nn.Linear(512, config.hidden_size//2)
        self.linear_2 = nn.Linear(config.hidden_size//2, config.hidden_size)
        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(config.hidden_size//2)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size)
        #self.kg_dim = config.kg_dim
        #self.kg_poll_size = config.kg_pool_size
    def forward(self, x):
        #channel first
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.d1(x)
        x = self.linear_2(x)
        x = self.layernorm_2(x)
        x = self.relu(x)
        x = self.d2(x)
        return x
    @property
    def proj_out_num(self):
        num = 1 #self.kg_dim // self.kg_poll_size
        return num


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type')

    if projector_type == 'linear':
        return ImageProjector(config)
    

    elif projector_type == 'spp':
        return SpatialPoolingProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)


    elif projector_type == 'identity':
        return IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
    
def build_kg_projector(config, delay_load=False, **kwargs):
    return KGProjector(config)
