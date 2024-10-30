import torch
from torch import nn
from backbone import RGBXTransformer

def print_tensor_shape(tensor, name):
    print(f'{name} shape: {tensor.shape}')

def test_rgbx_transformer():
    # 配置参数
    in_chans = 1
    patch_size = 4
    dims = [96, 192, 384, 768]
    depths = [2, 2, 9, 2]
    downsample_version = "v1"
    d_state = 16
    drop_rate = 0.0
    drop_path_rate = 0.2
    upsample_option = False
    
    # 创建模型
    model = RGBXTransformer(
        in_chans=in_chans,
        patch_size=patch_size,
        dims=dims,
        depths=depths,
        downsample_version=downsample_version,
        d_state=d_state,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建测试输入
    input_tensor1 = torch.randn(1, 1, 480, 640).to(device)  # 假设输入是 224x224 的 RGB 图像
    input_tensor2 = torch.randn(1, 1, 480, 640).to(device)
    print_tensor_shape(input_tensor1, 'Input Tensor1')
    print_tensor_shape(input_tensor2, 'Input Tensor2')

    feature_maps1, feature_maps2, output_tensor = model(input_tensor1, input_tensor2)
    print_tensor_shape(output_tensor, 'Output Tensor')
    print("rgb:")
    for i, fmap in enumerate(feature_maps1):
        print_tensor_shape(fmap, f'Feature Map {i}')
    print("ir:")
    for i, fmap in enumerate(feature_maps2):
        print_tensor_shape(fmap, f'Feature Map {i}')

if __name__ == '__main__':
    test_rgbx_transformer()
