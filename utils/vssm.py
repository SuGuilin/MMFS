from torch import nn
import torch
from collections import OrderedDict
import math
from timm.models.layers import trunc_normal_

from .utils import *
from .modules import VSSBlock


class VSSM(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # =========================
        d_state=16,
        # =========================
        drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        downsample_version: str = "v2",
        upsample_option=False,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.up = upsample_option
        self.d_state = d_state if d_state is not None else math.ceil(dims[0] / 6)

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        # [start, end, steps]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # In order to make both the encoder and decoder reuse VSSM, upsampling and downsampling can only be selected.
            if upsample_option:
                # At this time, the dims and depths list is the default value reversed
                sample = (
                    nn.Sequential(
                        nn.Conv2d(dims[i_layer - 1] * 2, dims[i_layer - 1], 3, 1, 1),
                        Permute(0, 2, 3, 1),
                        UpsampleExpand(dim=dims[i_layer], norm_layer=norm_layer),
                    )
                    if (i_layer != 0)
                    else nn.Identity()
                )
                # sample = Upsample_conv(dim=dims[i_layer], norm_layer=norm_layer) if (i_layer != 0) else nn.Identity()
            elif downsample_version == "v2":
                sample = (
                    self._make_downsample(
                        self.dims[i_layer],
                        self.dims[i_layer + 1],
                        norm_layer=norm_layer,
                    )
                    if (i_layer < self.num_layers - 1)
                    else nn.Identity()
                )
            else:
                sample = (
                    PatchMerging2D(
                        self.dims[i_layer],
                        self.dims[i_layer + 1],
                        norm_layer=norm_layer,
                    )
                    if (i_layer < self.num_layers - 1)
                    else nn.Identity()
                )

            self.layers.append(
                self._make_layer(
                    dim=self.dims[i_layer],
                    depth=depths[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    sample=sample,
                    d_state=self.d_state,
                    drop_rate=drop_rate,
                    up=upsample_option,
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            # nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            nn.Conv2d(
                dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=48,
        depth=2,
        drop_path=[0.1, 0.1],
        norm_layer=nn.LayerNorm,
        sample=nn.Identity(),
        d_state=16,
        drop_rate=0.0,
        up=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    d_state=d_state,
                    drop=drop_rate,
                    **kwargs,
                )
            )
        blocks.append(Permute(0, 3, 1, 2))
        blocks.append(nn.Conv2d(dim, dim, 3, 1, 1))
        blocks.append(Permute(0, 2, 3, 1))
        if up:
            return nn.Sequential(
                OrderedDict(
                    sample=sample,
                    blocks=nn.Sequential(
                        *blocks,
                    ),
                )
            )

        return nn.Sequential(
            OrderedDict(
                blocks=nn.Sequential(
                    *blocks,
                ),
                sample=sample,
            )
        )

    def forward_features(self, modal):
        outs = []
        modal = self.patch_embed(modal)
        modal = self.pos_drop(modal)
        for layer in self.layers:
            blocks_module, sample_moudle = layer[0], layer[1]
            outs.append(modal)
            # modal = layer(modal)
            # print("modal:", modal.shape)
            # print("blocks_modal:",blocks_module(modal).shape)
            modal = blocks_module(modal) + modal
            modal = sample_moudle(modal)
        return modal, outs

    def forward_features_up(self, modal, fused_outs):
        for idx, layer_up in enumerate(self.layers):
            sample_moudle_up, blocks_module_up = layer_up[0], layer_up[1]
            if idx == 0:
                modal = blocks_module_up(modal) + modal
                modal = sample_moudle_up(modal)
                # modal = layer_up(modal)
            else:
                # modal = layer_up(modal + fused_outs[-idx])
                # print(modal.shape)
                # print(fused_outs[-idx].shape)
                modal = torch.cat([modal, fused_outs[-idx]], dim=3)
                # print(modal.shape)
                # print("modal:", modal.shape)
                # print("blocks_modal:",blocks_module_up(modal).shape)
                modal = sample_moudle_up(modal.permute(0, 3, 1, 2).contiguous())
                # print(modal.shape)
                modal = blocks_module_up(modal) + modal

        return modal

    def forward(self, modal: torch.Tensor, fused_outs=None):
        if self.up:
            assert (
                fused_outs is not None
            ), "fused_outs cannot be None when upsample_option is True"
            modal = self.forward_features_up(modal, fused_outs)
            return modal
        else:
            modal, outs = self.forward_features(modal)
            return modal, outs
