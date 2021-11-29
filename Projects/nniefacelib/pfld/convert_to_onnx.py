"""
This code is used to convert the pytorch models into an onnx format models.
"""
import torch.onnx
# from pfld.pfld import PFLDInference

# from pfld.PFLD_Ultralight import PFLDInference

from pfld.PFLD_Ultralight_Slim import PFLDInference



input_img_size = 112  # define input size

# model_path = "models/pretrained/checkpoint_epoch_final.pth"

# model_path = "models/pretrained/checkpoint_robust.pth"


#model slim 0.5


# model of slim :0.25 

model_path = "/media/jcq/Soft/Pytorch/nniefacelib/PFPLD/models/checkpoint/model-Slim/checkpoint_epoch_24.pth"


checkpoint = torch.load(model_path)
net = PFLDInference()
net.load_state_dict(checkpoint)
net.eval()
net.to("cuda")

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"

dummy_input = torch.randn(1, 3, 112, 112).to("cuda")

torch.onnx.export(net, dummy_input, model_path, export_params=True, 
verbose=False, input_names=['input'], output_names=['pose', 'landms'])


# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#                                input_names=input_names, output_names=output_names)