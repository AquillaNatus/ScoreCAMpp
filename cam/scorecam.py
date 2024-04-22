import torch
import torch.nn.functional as f
from cam.basecam import *
from utils import *

class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        logit = f.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        # activations = reshape_transform(activations)
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

            # upsampling
            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
            saliency_map = f.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            
            if saliency_map.max() == saliency_map.min():
              continue
            
            # normalize to 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            # how much increase if keeping the highlighted region
            # predication on masked input
          #   print(f'Size of input is: {input.shape}') #(1,3,224,224)
          #   print(f'Size of norm_saliency map: {norm_saliency_map.shape}') #(1,1,224,224)
            output = self.model_arch(input * norm_saliency_map)
            output = f.softmax(output)
            score = output[0][predicted_class]
            # print(f'Score is: {score}')   # Score is: tensor([0.0057], device='cuda:0')
            score_saliency_map +=  score * saliency_map
            # print(f'Score_saliency map: {score_saliency_map}')
        score_saliency_map = f.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)