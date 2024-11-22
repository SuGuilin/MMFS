from torch import nn
import torch
from einops import rearrange
import numbers
import torch.nn.functional as F

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3, bias=False)
        # self.proj1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        # self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x = self.lrelu(self.proj1(x))
        # x = self.lrelu(self.proj2(x)).permute(0, 2, 3, 1)
        x = self.proj1(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):
    def __init__(
        self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B C H W
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x



class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)
        return x
    
class UpsampleExpand(nn.Module):
    def __init__(self, dim, scale_factor=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)
        self.norm = norm_layer(dim * scale_factor)
    
    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.linear(x).permute(0, 3, 1, 2).contiguous() # B, C/2, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C/2
        x = self.norm(x)
        return x

class Upsample_conv(nn.Module):
    def __init__(self, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.convt = nn.Sequential(
        #     nn.ConvTranspose2d(dim*2, dim, 4, 2, 1, bias=False), 
        #     nn.ReLU()
        # )
        self.conv = nn.Conv2d(dim*2, dim*4, 3, 1, 1) #nn.Linear(dim*2, dim, bias=False)
        self.PixelShuffle = nn.PixelShuffle(2)
        self.norm = norm_layer(dim)
    def forward(self, x):
        """
        x: B, H, W, C
        """
        # print(x.shape)
        x = self.conv(x.permute(0, 3, 1, 2).contiguous()) # B, C/2, H, W
        # print(x.shape)
        x = self.PixelShuffle(x).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C/2
        
        # x = self.convt(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # print(x.shape)
        x = self.norm(x)
        return x

class Final_UpsampleExpand(nn.Module):
    def __init__(self, dim, scale_factor=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
    
    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C/2
        x = self.norm(x)
        return x
        

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x) # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size,
                      c=C // (self.patch_size ** 2))
        x = self.norm(x)
        return x


class FinalUpsample_X4(nn.Module):
    def __init__(self, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    
    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous() # B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous() # B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 4H, 4W, C
        x = self.norm(x)
        return x

class Upsample_X4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_X4, self).__init__()
        self.upsample = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels // 4, in_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels*16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.PixelShuffle(4),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.upsample(x)

#######################################################################

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


