import torch
import torch.nn.functional as F
import torchvision.models as models
import warnings
from utils import *
from cam.scorecam_x import *
# from cam.scorecam_x_prev import *
from cam.scorecam import *
from cam.layercam import *
from cam.other_cams import *
# from cam.scorecam_d import *
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# img_list_path = ['images/tower.JPEG','images/dogsled.jpg','images/tigercat.jpg']
ROOT_NAME = 'Example_Images'
# ROOT_NAME = 'A_images'
img_list_path = os.listdir(ROOT_NAME)
for idx, item in enumerate(img_list_path):
  img_list_path[idx] = ROOT_NAME + '/' + img_list_path[idx]
# print(img_list_path)

dir_name = ['scorecam', 'scorecam_pp', 'layercam', 'gradcam', 'gradcampp', 'xgradcam', 'Augmented GradCAM']
# dir_name = ['Aug']
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
vgg = models.vgg19(pretrained=True).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_35',input_size=(224, 224))

vgg_scorecam = ScoreCAM(vgg_model_dict)
vgg_scorecam_x = ScoreCAM_x(vgg_model_dict)
vgg_layercam = LayerCAM(vgg_model_dict)
vgg_xgradcam = XGradCam(vgg_model_dict)
vgg_gradcampp = GradCAMpp(vgg_model_dict)
vgg_gradcam = GradCAM(vgg_model_dict)
vgg_augpp = aug_GradCAMpp(vgg_model_dict)
# vgg_model_list = [vgg_augpp]
vgg_model_list = [vgg_scorecam, vgg_scorecam_x, vgg_layercam, vgg_gradcampp, vgg_gradcam, vgg_xgradcam, vgg_augpp]
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

resnet = models.resnet18(pretrained=True).eval()
resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))

resnet_scorecam = ScoreCAM(resnet_model_dict)
resnet_scorecam_x = ScoreCAM_x(resnet_model_dict)
resnet_layercam = LayerCAM(resnet_model_dict)
resnet_xgradcam = XGradCam(resnet_model_dict)
resnet_gradcampp = GradCAMpp(resnet_model_dict)
resnet_gradcam = GradCAM(resnet_model_dict)
resnet_augpp = aug_GradCAMpp(resnet_model_dict)
# resnet_model_list = [resnet_augpp]
resnet_model_list = [resnet_scorecam, resnet_scorecam_x, resnet_layercam, resnet_gradcampp, resnet_gradcam,resnet_xgradcam,resnet_augpp]
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

for idx, cam_model in enumerate(vgg_model_list):
  for index, img_path in enumerate(img_list_path):
    if index == 1:
      break
    print(f'|____{idx}/{index}/{len(img_list_path)}')
    input_image = load_image(img_path)
    input_ = apply_transforms(input_image)
    if torch.cuda.is_available():
      input_ = input_.cuda()
    saliency_map = cam_model(input_)
    if saliency_map == None:
      continue
    if not os.path.exists(os.path.join('cam_images/vgg/' + dir_name[idx])):
      os.mkdir(os.path.join('cam_images/vgg/' + dir_name[idx]))
      
    path1 = os.path.join('cam_images/vgg/'+dir_name[idx]+'/img_' + str(index) +'.png')
    # path2 = os.path.join('cam_images/vgg/'+dir_name[idx]+'/img_' + str(index) +'.pdf')
    # print(path1)
    print(f'vgg//{idx}//{index}')
    visualize_(input_.cpu(),saliency_map.type(torch.FloatTensor).cpu(),path1)
    # visualize_(input_.cpu(),saliency_map.type(torch.FloatTensor).cpu(),path2)



for idx, cam_model in enumerate(resnet_model_list):
  # print(idx)
  for index, img_path in enumerate(img_list_path):
    # print(f'|___{index}')
    input_image = load_image(img_path)
    input_ = apply_transforms(input_image)
    if torch.cuda.is_available():
      input_ = input_.cuda()
      
    saliency_map = cam_model(input_)
    if saliency_map == None:
      continue
    if not os.path.exists(os.path.join('cam_images/resnet/' + dir_name[idx])):
      os.mkdir(os.path.join('cam_images/resnet/' + dir_name[idx]))
    path1 = os.path.join('cam_images/resnet'+'/'+dir_name[idx]+'/'+'img_' + str(index) + '.png')
    # path2 = os.path.join('cam_images/resnet'+'/'+dir_name[idx]+'/'+'img_' + str(index) + '.pdf')    
    # print(path2)
    print(f'resnet//{idx}//{index}')
    visualize_(input_.cpu(),saliency_map.type(torch.FloatTensor).cpu(),path1)
    # visualize_(input_.cpu(),saliency_map.type(torch.FloatTensor).cpu(),path2)

print('Resnet Completed')
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
