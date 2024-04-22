from cam.basecam import *
import torch.nn.functional as f

class LayerCAM(BaseCAM):

    def __init__(self, model_dict):
        super().__init__(model_dict)
        self.predicted_confidence_cam_list = []
        self.avg_drop = 0
        self.avg_increase = 0
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

        #logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class = predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        one_hot_output = torch.FloatTensor(1, logit.size()[-1]).zero_()
        one_hot_output[0][predicted_class] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        self.model_arch.zero_grad()
        # Backward pass with specified target
        logit.backward(gradient=one_hot_output, retain_graph=True)
        activations = self.activations['value'].clone().detach()
        gradients = self.gradients['value'].clone().detach()
        b, k, u, v = activations.size()

        with torch.no_grad():
            activation_maps = activations * f.relu(gradients)
            cam = torch.sum(activation_maps, dim=1).unsqueeze(0)
            cam = f.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_min == cam_max:
              return None
            norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
#
        return norm_cam

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