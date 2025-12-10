# -*- coding: utf-8 -*-
import torch
import torch.onnx
import os
# 硬编码字符表
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']

from model.LPRNet import build_lprnet

def export_onnx():
    img_size = [94, 24]
    lpr_max_len = 8
    dropout_rate = 0
    class_num = len(CHARS)
    weights_path = './weights/Final_LPRNet_model.pth'
    output_onnx = 'LPRNet.onnx'

    print(f"PyTorch Version: {torch.__version__}")

    lprnet = build_lprnet(lpr_max_len=lpr_max_len, phase=False, class_num=class_num, dropout_rate=dropout_rate)

    device = torch.device('cpu')
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        lprnet.load_state_dict(new_state_dict)
    else:
        print("Error: Weights not found")
        return

    lprnet.eval()

    # 输入形状：Batch=1, C=3, H=24, W=94
    dummy_input = torch.randn(1, 3, 24, 94)

    print(f"正在以 Opset 11 导出为 {output_onnx} ...")

    # 核心修改：强制 Opset 11，且关闭 Constant Folding (防止Mac崩)
    torch.onnx.export(lprnet,
                      dummy_input,
                      output_onnx,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      do_constant_folding=True, # 必须为 False 才能在 Mac 上跑通 Opset 11
                      opset_version=11)          # 必须为 11 才能让 OpenCV 读懂

    print("✅ 导出完成！")

if __name__ == '__main__':
    export_onnx()
