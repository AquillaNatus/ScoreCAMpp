import torch
import torch.nn.functional as f
from cam.basecam import *
from utils import *

class ScoreCAM_x(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict, threshold=0.5, isthreshold=True):
        super().__init__(model_dict)
        self.threshold = threshold
        self.predicted_confidence_cam = 0
        self.predicted_confidence_cam_list = []
        self.avg_drop = 0
        self.avg_increase = 0
        self.isthreshold = isthreshold
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

            """
            saliency_map = f.tanh(saliency_map)
    
            if self.isthreshold:
                saliency_map = torch.where(saliency_map < self.threshold, 0, saliency_map)

            norm_saliency_map = saliency_map.clone()
            """
            
            saliency_map = f.sigmoid(f.relu(saliency_map/2))
            norm_saliency_map = saliency_map.clone()
            # norm_saliency_map = f.sigmoid(norm_saliency_map)

           
            # if self.isthreshold:
            #     norm_saliency_map = torch.where(norm_saliency_map<self.threshold,0,saliency_map)
            
            
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