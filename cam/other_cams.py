import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from utils import *
import torch.nn.functional as f
from statistics import mode, mean

class _BaseCAM(object):
    """ Base class for Class activation mapping.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """

    def __init__(self, model_dict):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']

        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        if torch.cuda.is_available():
          self.model_arch.cuda()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value'] = grad_output[0].cuda()
            else:
              self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
              self.activations['value'] = output.cuda()
            else:
              self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            self.target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            self.target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            self.target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            self.target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=True):
        return None

    def __call__(self, input, class_idx=None, retain_graph=True):
        return self.forward(input, class_idx, retain_graph)

    def get_output(self, x):
      for name, module in self.model_arch._modules.items():
            x = module(x)
      return x

class XGradCam(_BaseCAM):
    def __init__(self, model_dict,use_cuda=True):
        super().__init__(model_dict)
        self.model = model_dict['arch']
        self.model.eval()
        self.cuda = use_cuda
        self.predicted_confidence_cam_list = []
        self.predicted_confidence_cam = 0
        self.avg_drop = 0
        self.avg_increase = 0
        if self.cuda:
            self.model = self.model.cuda()

        # self.extractor = ModelOutputs(model_dict)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=-1):
        output = self.model(input)
        # for name, module in self.model._modules.items():
        #     output = module(output)

        # output = output.view(output.size(0), -1)
        # output = self.model.classifier(output)
        # output.backward()
        features = self.activations['value']
        if self.cuda:
            features = features.cuda()
            output = output.cuda()


        if index == -1:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients['value'].cpu().data.numpy()

        target = features
        target = target.cpu().data.numpy()[0, :]

        # XGrad_CAM
        X_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))
        X_weights = X_weights / (np.sum(target, axis=(1, 2)) + 1e-6)
        # Grad_CAM
        weights = np.mean(grads_val, axis=(2, 3))[0, :]

        X_cam = np.zeros(target.shape[1:], dtype=np.float32)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            X_cam += X_weights[i] * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        X_cam = np.maximum(X_cam, 0)
        X_cam = cv2.resize(X_cam, (224, 224))
        if X_cam.any() == None:
          return None
        if np.min(X_cam) == np.max(X_cam):
          return None
        X_cam = X_cam - np.min(X_cam)
        X_cam = X_cam / np.max(X_cam)
        X_cam = torch.unsqueeze(torch.from_numpy(X_cam), 0)
        X_cam = torch.unsqueeze(X_cam, 0)
        print(X_cam.size())
        return  X_cam

    def metrics(self,model,input_):

        predicted_confidence, predicted_class = model(input_).max(1)
        predicted_class = predicted_class.item()
        predicted_confidence = predicted_confidence.item()

        scorecam_map = self(input_)
        if scorecam_map == None:
            self.predicted_confidence_cam = 0
        else:
            scorecam_map =  torch.from_numpy(scorecam_map).cuda()
            scorecam_map_  = scorecam_map * input_
            scorecam_map_output = model(scorecam_map_)
            predicted_confidence_cam = scorecam_map_output[0][predicted_class]
            self.predicted_confidence_cam = predicted_confidence_cam.item()

        self.predicted_confidence_cam_list.append(self.predicted_confidence_cam)
        if predicted_confidence > self.predicted_confidence_cam:
          self.avg_drop = self.avg_drop + (predicted_confidence - self.predicted_confidence_cam)/predicted_confidence
        else:
          self.avg_increase = self.avg_increase + 1

    def percentize(self, count):
        self.avg_drop = self.avg_drop*100/count
        self.avg_increase = self.avg_increase*100/count

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)

        prob = f.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            # print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = f.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model_dict):
        model = model_dict['arch'].cuda()
        model_type = model_dict['type']
        self.model_arch = model_dict['arch']
        self.predicted_confidence_cam_list = []
        self.predicted_confidence_cam = 0
        self.avg_drop = 0
        self.avg_increase = 0
        layer_name = model_dict['layer_name']
        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        _, _, h, w = x.size()

        # object classification
        score = self.model(x)

        prob = f.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()

        cam = self.getGradCAMpp(self.values, score, idx)
        if cam == None:
          return None
        saliency_map = f.interpolate(cam,size=(h,w), mode='bilinear', align_corners=False)
        return saliency_map

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''
        # print(("Printing from GradCAMpp->getGradCAMpp function:").upper())
        self.model.zero_grad()

        # score[0, idx].backward(retain_graph=True)
        score[0, idx].backward(retain_graph=False)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = f.relu(score[0, idx].exp() * gradients)
        # print(f'Alpha is: {alpha}')
        # print(f'Gradients are: {gradients}')
        # print(f'Relu_grad are: {relu_grad}')
        # print(f'Score exp is: {score[0,idx].exp()}')
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)
        # print(f"Weights are: {weights}")
        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = f.relu(cam)
        # print(cam)
        # print(f'Max and min of cam from GradCAMpp are: {torch.max(cam)} and {torch.min(cam)}')
        if cam.any() == None:
          return None
        if torch.min(cam) == torch.max(cam):
          return None
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data

    def metrics(self,model,input_):

        predicted_confidence, predicted_class = model(input_).max(1)
        predicted_class = predicted_class.item()
        predicted_confidence = predicted_confidence.item()

        scorecam_map = self.forward(input_)
        if scorecam_map == None:
            self.predicted_confidence_cam = 0
        else:
            scorecam_map_  = scorecam_map * input_
            scorecam_map_output = model(scorecam_map_)
            predicted_confidence_cam = scorecam_map_output[0][predicted_class]
            self.predicted_confidence_cam = predicted_confidence_cam.item()

        self.predicted_confidence_cam_list.append(self.predicted_confidence_cam)
        if predicted_confidence > self.predicted_confidence_cam:
          self.avg_drop = self.avg_drop + (predicted_confidence - self.predicted_confidence_cam)/predicted_confidence
        else:
          self.avg_increase = self.avg_increase + 1

    def percentize(self, count):
        self.avg_drop = self.avg_drop*100/count
        self.avg_increase = self.avg_increase*100/count

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model_dict):
        model = model_dict['arch'].cuda()
        model_type = model_dict['type']
        self.model_arch = model_dict['arch']
        self.predicted_confidence_cam = 0
        self.predicted_confidence_cam_list = []
        self.avg_drop = 0
        self.avg_increase = 0

        layer_name = model_dict['layer_name']
        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        _,_,h,w = x.size()

        score = self.model(x)
#         print(f'Input Size: {x.size()}')
        prob = f.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
#             print("predicted class ids {}\t probability {}".format(idx, prob))

        # calculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx) # Size of CAM: torch.Size([1, 1, 12, 16])
        """
        saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
        saliency_map = f.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        """
        if cam == None:
          return None
        saliency_map = f.interpolate(cam,size=(h,w), mode='bilinear', align_corners=False)
        return saliency_map

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        # score[0, idx].backward(retain_graph=True)
        score[0, idx].backward(retain_graph=False)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = f.relu(cam)
        if cam.any() == None:
          return None

        if torch.min(cam) == torch.max(cam):
          return None
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        return cam.data

    def metrics(self,model,input_):

        predicted_confidence, predicted_class = model(input_).max(1)
        predicted_class = predicted_class.item()
        predicted_confidence = predicted_confidence.item()

        scorecam_map = self.forward(input_)
        if scorecam_map == None:
            self.predicted_confidence_cam = 0
        else:
            scorecam_map_  = scorecam_map * input_
            scorecam_map_output = model(scorecam_map_)
            predicted_confidence_cam = scorecam_map_output[0][predicted_class]
            self.predicted_confidence_cam = predicted_confidence_cam.item()

        self.predicted_confidence_cam_list.append(self.predicted_confidence_cam)
        if predicted_confidence > self.predicted_confidence_cam:
          self.avg_drop = self.avg_drop + (predicted_confidence - self.predicted_confidence_cam)/predicted_confidence
        else:
          self.avg_increase = self.avg_increase + 1

    def percentize(self, count):
        self.avg_drop = self.avg_drop*100/count
        self.avg_increase = self.avg_increase*100/count

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



class aug_GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model_dict):
        model = model_dict['arch'].cuda()
        model_type = model_dict['type']
        self.model_arch = model_dict['arch']
        self.predicted_confidence_cam_list = []
        self.predicted_confidence_cam = 0
        self.avg_drop = 0
        self.avg_increase = 0
        layer_name = model_dict['layer_name']
        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, r=50,idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        _, _, h, w = x.size()
        score = self.model(x)

        prob = f.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
        # caluculate cam of the predicted class
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=45, translate=(0.2, 0.2)),
        ])
        
        cam = self.getGradCAMpp(self.values, score, idx)
        # object classification
        for idx in range(r):
            x_ = transform(x)
            score = self.model(x_)
            prob = f.softmax(score, dim=1)
            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                prob = prob.item()
            # caluculate cam of the predicted class
            cam_ = self.getGradCAMpp(self.values, score, idx)
            if cam_ is None:
                continue
            else:
                cam = cam + cam_
                
        if cam == None:
          return None
    
        saliency_map = f.interpolate(cam,size=(h,w), mode='bilinear', align_corners=False)
        saliency_map = (saliency_map - saliency_map.min())/(saliency_map.max()-saliency_map.min())
        return saliency_map


    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''
        # print(("Printing from GradCAMpp->getGradCAMpp function:").upper())
        self.model.zero_grad()

        # score[0, idx].backward(retain_graph=True)
        score[0, idx].backward(retain_graph=False)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)
        relu_grad = f.relu(score[0, idx].exp() * gradients)  
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)
        cam = (weights * activations).sum(1, keepdim=True)
        cam = f.relu(cam)
        if cam.any() == None:
          return None
        if torch.min(cam) == torch.max(cam):
          return None
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data