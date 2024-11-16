import sys
import os

sys.path.append("/home/suguilin/MMFS/")
sys.path.append("/home/suguilin/VMamba/")
import torch
import torch.nn as nn
from encoder.dense_mamba import DenseMamba, DenseMambaBlock
from encoder.auxiliary_net import AuxiliaryNet
from encoder.mixmap_net import CombineNet
from encoder.Mixture_Experts import MoLE, MoGE, MambaMoE
from encoder.degradetion import DegradationModel
from encoder.textencoder import TextEncoder
from encoder.cross_mamba import CrossMamba, Fusion_Embed, resize, CrossAttention
from encoder.cross_module import CrossBlock
from encoder.cross_fusion import Chanel_Cross_Attention
from decoder.seghead import DecoderHead
from utils.utils import Final_UpsampleExpand, FinalUpsample_X4, Permute, Upsample_X4, LayerNorm
from utils.modules import Restormer_CNN_block
from utils.NAFBlock import NAFBlock
from utils.MixBlock import MixBlock
from memory_profiler import profile
from classification.models.vmamba import Backbone_VSSM


class MMoEFusion(nn.Module):
    def __init__(
        self,
        device,
        num_classes=9,
        embed_dim=768, #384,
        patch_size=4,
        in_chans=1,
        out_chans=3,
        # depths=[2, 2, 4],
        depths=[1, 2, 2],
        dims=[96, 96, 96],#[48, 48, 48], [64, 64, 64]
        channels=[96, 192, 384, 768], #[48, 96, 192, 384], [64, 128, 256, 512]
        d_state=16,
        drop_rate=0.0,
        drop_path_rate=0.2,
        seg_norm=nn.BatchNorm2d,
        align_corners=False,
        norm_layer=nn.LayerNorm,
        num_experts=4,
    ):
        super(MMoEFusion, self).__init__()
        # self.degradation = DegradationModel()
        # self.backbone = DenseMamba(
        #     in_chans=in_chans,
        #     d_state=d_state,
        #     patch_size=patch_size,
        #     depths=depths,
        #     dim=dims[-1],
        #     norm_layer=norm_layer,
        #     drop_rate=drop_rate,
        #     drop_path_rate=drop_path_rate,
        #     num_experts=num_experts,
        # )
        channel_first = True
        self.backbone = Backbone_VSSM(
            pretrained='/home/suguilin/MMFS/pretrained/classification/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            # pretrained=None,
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
        )
        # 直接读入文本编码npy了
        # self.textencoder = TextEncoder(device=device) 
        # self.cross_mamba_rgb = CrossMamba()
        # self.cross_mamba_ir = CrossMamba()
        self.fusion_embed = Fusion_Embed(embed_dim=channels, norm_layer=norm_layer)
        self.dim_match = nn.Conv2d(in_channels=embed_dim, out_channels=dims[-1], kernel_size=1)
        self.seg_fus = Chanel_Cross_Attention(dim=dims[-1], num_head=8, bias=True)
        self.dm_fusion = nn.Sequential(
            Permute(0, 2, 3, 1),
            DenseMambaBlock(dims=dims[-1]),
            Permute(0, 3, 1, 2),
        )
        # self.expand = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=dims[-1]*2, out_channels=dims[-1]*2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(dims[-1]*2, dims[-1]*2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(dims[-1]*2, dims[-1], kernel_size=1),
        # )
        
        # self.cross_fusion = CrossBlock(modals=2, hidden_dim=32)
        # self.auxiliary_net = AuxiliaryNet(dim=dims[-1])
        # self.mixmap_net = CombineNet(num_heads=8, d_model=64, dim=dims[-1])
        #self.mole = MoLE(dim=dims[-1], num_experts=num_experts, top_k=1)
        # self.local_expand = nn.ConvTranspose2d(in_channels=dims[-1], out_channels=dims[-1], kernel_size=4, stride=2, padding=1)
        # self.local_norm = nn.LayerNorm(dims[-1])
        # self.cross_attention = CrossAttention(embed_dim=768, num_heads=8)
        # self.refine_expert = NAFBlock(c=dims[-1])
        # self.moge = MoGE(dim=dims[-1], num_experts=num_experts, top_k=2)
        # self.global_incre = nn.Conv2d(dims[-1], dims[-1] * 2, kernel_size=1)
        # self.global_norm = nn.LayerNorm(dims[-1] * 2)
        # self.mixer = nn.Identity()

        # self.down_rgb = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=2, padding=1)
        # self.down_ir = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=2, padding=1)

        # self.final_up = FinalUpsample_X4(
        #     dim=dims[-1], patch_size=4, norm_layer=norm_layer
        # )
        self.channels = channels
        self.align_corners = align_corners
        self.seghead = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=seg_norm, embed_dim=embed_dim)
        self.conv_after_body = nn.Conv2d(dims[-1] * 2, dims[-1], 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], 3, 1, 1), 
            nn.LeakyReLU(inplace=True)
        )
        # self.final_up = Upsample_X4(dims[-1] * 2, out_chans)
        self.final_up = Upsample_X4(dims[-1], out_chans)
        # self.decoder = nn.Conv2d(dims[-1] // 4, out_chans, 3, 1, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(3 * 3, dims[-1], 1),
            # nn.Conv2d(dims[-1] * 3, dims[-1] // 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(dims[-1] // 2, dims[-1] // 4, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(dims[-1] // 4, out_chans, 1),
            nn.Conv2d(dims[-1], 3, kernel_size=1)
        )

        # helper methods needed for reconstruction as follows:

        # self.rec_rgb = Restormer_CNN_block(in_dim=dims[-1] * 4, out_dim=dims[-1])
        # self.rec_ir = Restormer_CNN_block(in_dim=dims[-1] * 4, out_dim=dims[-1])

        # self.final_up_1 = FinalUpsample_X4(
        #     dim=dims[-1], patch_size=4, norm_layer=norm_layer
        # )

        # self.decoder_1 = nn.Sequential(
        #     Permute(0, 3, 1, 2),
        #     nn.Conv2d(dims[-1], dims[-1] // 2, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(dims[-1] // 2, dims[-1] // 4, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(dims[-1] // 4, out_chans*3, 3, 1, 1),
        # )

    def forward(self, rgb, ir, text_rgb, text_ir):
        # print("before:", rgb.shape, ir.shape)
        input_size = rgb.shape
        # rgb = self.degradation(rgb)
        # # deg_rgb = rgb
        # ir = self.degradation(ir)
        # deg_ir = ir
        # print("after:", rgb.shape, ir.shape)
        # outs_rgb, outs_ir, branch_rgb, branch_ir, loss_aux = self.backbone(rgb, ir)
        loss_aux = 0
        outs_rgb = self.backbone(rgb)
        outs_ir = self.backbone(ir)
        B, C, H, W = outs_rgb[0].shape
        # print("dense:", rgb_dense.shape, ir_dense.shape)
        # _, _, bert_rgb = self.textencoder(text_rgb)
        # print(bert_rgb.shape)
        # _, _, bert_ir = self.textencoder(text_ir)
        # print(bert_ir.shape)
        bert_rgb = text_rgb
        bert_ir = text_ir
        #cross_rgb = self.cross_mamba_rgb(branch_rgb[1:3], bert_rgb)
        #cross_ir = self.cross_mamba_ir(branch_ir[1:3], bert_ir)
        semanfea_fusion = self.fusion_embed(outs_rgb, outs_ir)
    
        # print(cross_rgb.shape)
        # print(cross_ir.shape
        ######
        #cross_img = self.cross_fusion(cross_rgb, cross_ir, H, W)
        # print(cross_img.shape)
        # self.mixer = MixBlock(
        #     input_resolution=(H, W),
        #     num_heads=4,
        #     d_model=32,
        # ).to(rgb.device)

        # self.MMoE = MambaMoE(
        #     W,
        #     H,
        #     C,
        #     C,
        #     local_num_experts=4,
        #     gobal_num_experts=4,
        #     noisy_gating=True,
        #     local_k=2,
        #     gobal_k=2,
        # ).to(rgb.device)

        # reconstructions

        # rgb_rec = self.rec_rgb(rgb_dense).permute(0, 2, 3, 1)
        # ir_rec = self.rec_ir(ir_dense).permute(0, 2, 3, 1)

        # rgb_rec = self.final_up(rgb_rec)
        # ir_rec = self.final_up_1(ir_rec)

        # rgb_rec = self.decoder_1(rgb_rec)
        # ir_rec = self.decoder(ir_rec)
        # return rgb_rec, ir_rec

        # Att = self.auxiliary_net(rgb, ir)
        # Att = self.mixmap_net(rgb, ir)
        # rgb_enc = rgb_enc * Att
        # ir_enc = ir_enc * Att
        # y_local, rgb_local, ir_local = self.mole(rgb_enc, ir_enc, rgb_dense, ir_dense)
        # rgb_enc = self.down_rgb(rgb_enc)
        # ir_enc = self.down_ir(ir_enc)

        # y_local = self.mole(branch_rgb[0][:, 0:32, :, :], branch_ir[0][:, 0:32, :, :])
        #######
        # y_local = self.mole(branch_rgb[0], branch_ir[0])
        #######
        '''
        # 添加局部专家结果与文本的交叉注意力
        y_local = y_local.contiguous().view(y_local.size(0), y_local.size().numel() // y_local.size(0) // 768, 768)
        text_fea = torch.cat([text_rgb, text_ir], dim=2).view(B, -1, 768)
        local_cross_att = self.cross_attention(text_fea, y_local, y_local)
        local_cross_att = torch.nn.functional.adaptive_avg_pool1d(local_cross_att.permute(0, 2, 1), 1).permute(0, 2, 1)
        y_local = (y_local * local_cross_att).view(B, -1, H, W)
        '''
        # print(y_local.shape)

        # y_local = self.refine_expert(y_local)
        # rgb_local = self.refine_expert(rgb_local)
        # ir_local = self.refine_expert(ir_local)
        # y_global = self.moge(y_local, rgb_local, ir_local)
        # y_global = self.moge(y_local)
        # y_global = self.moge(y_local, rgb_enc, ir_enc)
        
        ######
        # y_global = self.moge(y_local, y_local, cross_img)
        ######
        # head_in.append(self.local_norm(self.local_expand(y_local).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        # head_in.append(self.global_norm(self.global_incre(y_global).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        # for i in range(len(head_in)):
        #     print(head_in[i].shape)

        # y_global, load_loss = self.MMoE(
        #     rgb_enc.view(B, -1), ir_enc.view(B, -1), rgb_dense, ir_dense
        # )

        # y_global = self.mixer(y_global.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # print(y_global.shape)
        # fused = self.final_up(y_global.permute(0, 2, 3, 1).contiguous())
        seg_out = self.seghead(semanfea_fusion)
        fused = semanfea_fusion[0]
        # fused = self.seg_fus(self.dim_match(_c), fused)
        # fused = self.dm_fusion(fused)

        seg_out = resize(input=seg_out, size=input_size[2:], mode='bilinear', align_corners=self.align_corners)
        # print(seg_out)
        # print("max_seg:", torch.max(seg_out))
        # print("min_seg:", torch.min(seg_out))
        # print(seg_out.shape)
        # fused = self.conv_after_body(torch.cat([y_global, y_local], dim=1)) + semanfea_fusion[0]
        
        # supplement = self.expand(semanfea_fusion[1])
        # fused = fused + supplement
        # fused = torch.cat([fused, supplement], dim=1)
        # fused = self.final_up(torch.cat([y_global, y_local], dim=1))
        fused = self.final_up(self.conv_before_upsample(fused))
        fused = self.decoder(torch.cat([fused, rgb, ir], dim=1))
        # print(fused.shape)
        # res = self.decoder(fused)
        return fused, seg_out, loss_aux#, deg_rgb, deg_ir #res#, load_loss


if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MMoEFusion(device=device).to(device)
    model.eval()
    rgb = torch.randn(2, 3, 288, 384).to(device)
    ir = torch.randn(2, 1, 288, 384).to(device)
    # text_rgb = [
    #     ['In this image with a resolution of 384X288, we can see a person walking down a street at night. The dense caption provides further details about the various objects present. A woman can be seen walking on the road, positioned between coordinates [97, 76, 171, 248]. Another person is also depicted walking, located at coordinates [256, 98, 276, 163]. A grey backpack can be observed on a person, positioned between coordinates [123, 110, 164, 173]. Moreover, a car is parked on the side of the road, situated at coordinates [63, 116, 111, 164]. Additionally, the region semantic provides additional information about different elements in the image. A woman is depicted walking with a suitcase between coordinates [0, 149, 383, 137]. There is also a black and white photo of a man standing in front of a wall, positioned at [341, 50, 42, 95]. Furthermore, a black object on a black background can be seen at coordinates [101, 169, 65, 45]. Additionally, a black and white photo of a building can be found, wherein the lights are on. It is located at [0, 97, 72, 33]. Lastly, a white figure is depicted against a black background, positioned at [124, 101, 38, 71].'], 
    #     ['In this 384x288 resolution image, a street scene unfolds before our eyes. Cars are parked in front of a building, creating a bustling urban atmosphere. A white car can be observed parked on the side of the road, specifically positioned between coordinates (272, 142) and (343, 185). Another car, described as being white, can also be found parked along the roadside, with its positioning spanning from (251, 148) to (278, 175). As we shift our attention towards the road, we notice the presence of white lines, stretching from (2, 186) to (185, 288). These lines guide the flow of traffic and add structure to the scene. Adjacent to the road, trees are visible on the sidewalk, occupying an area defined by the coordinates (1, 1) and (230, 175). Additional white lines grace the road, spanning from (1, 168) to (380, 288), delineating the various lanes for vehicles. Lastly, we can identify a sole white line adorning the road, positioned between coordinates (276, 208) and (371, 231). This image paints a vivid picture of an urban street, emphasizing the presence of cars, road markings, and greenery along the sidewalk.'], 
    # ]
    # text_ir = [
    #     ['In this 384X288 resolution image, the captivating scene unfolds at night as two people gracefully walk down a dimly lit street. The dense caption brings additional insight, revealing a person confidently strolling on the sidewalk, a woman in a white shirt nearby, and another person standing nearby. In the foreground, a white line appears, marking the side of the road. The region semantics provide a different perspective, describing the image as two people traversing a dark road at night, with a man standing against a black background. Additionally, a long black object with a long handle is present, while a black and white photo showcases a person standing on a hill. Adding to the intriguing composition, a black piano sits upon a black background. These elements coalesce within the image, inviting viewers to appreciate the beauty and serenity of nature blending seamlessly with human presence in the enigmatic night.'], 
    #     ['In this image, the resolution is 384X288, capturing a large building. The dense caption reveals various elements within the scene: a person can be seen walking on the sidewalk, while another stands in the distance. A long asphalt road stretches across the frame, accompanied by lush green grass on one side. Trees line the street, adding a touch of nature to the urban setting. Switching to the region semantic, a dark hallway comes into focus, with a person walking down it. Additionally, a green bush catches the eye, featuring a lighted area at its center. A single green leaf stands out against a black background. Finally, a picture of a gold and black square and a yellow light shining on a black background add visual interest to the overall composition.']
    # ]
    text_rgb = torch.randn(2, 256, 768).to(device)
    text_ir = torch.randn(2, 256, 768).to(device)
    # text_rgb = [
    # out, load_loss = model(rgb, ir, text_rgb, text_ir)
    # fused, seg_out, param_rgb, param_ir = model(rgb, ir, text_rgb, text_ir)
    fused, seg_out = model(rgb, ir, text_rgb, text_ir)

    print(fused.shape)
    print(seg_out.shape)
    # print(load_loss)
    # out1, out2 = model(rgb, ir)
    # print(out1.shape)
    # print(out2.shape)
