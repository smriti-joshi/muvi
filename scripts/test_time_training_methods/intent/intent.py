import torch
from torch import nn
from torch.nn import functional as G

# InTent implementation through a wrapper class
# no backpropagation
# combine predictions from different batch norm stats and combine them weighted by prediction entropy

########### helper functions ###########################

# Collects mean and variance of all layers and stores it in a list
def get_stats(net):
    mean, var = [], []
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mean.append(m.running_mean.clone().detach())
            var.append(m.running_var.clone().detach())
    return mean, var

def update_stats(net, mean, var):
    count = 0
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.running_mean = mean[count].clone().detach().cuda()
            m.running_var = var[count].clone().detach().cuda()
            count += 1
    return net

def make_prediction(net, imgs):
    with torch.no_grad():
        pred = net(imgs)
        pred_prob = torch.sigmoid(pred)

        e = softmax_entropy(pred_prob, mode='binary')
        confidence = 1 - e 
        return pred, confidence
    
def softmax_entropy(x, mode='standard', full=True):
    if mode == 'binary':
        ret = -x*torch.log2(x)-(1-x)*torch.log2(1-x)
        ret[x==0] = 0
        ret[x==1] = 0
    elif mode == 'standard':
        ret = -x*torch.log(x)-(1-x)*torch.log(1-x)
        ret[x==0] = 0
        ret[x==1] = 0

    if full:
        return ret
    else:
        return ret.mean()

########### main class #####################

class InTent(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.original_mean, self.original_variance = get_stats(self.model)

    def forward(self, x):

        outputs = self.forward_and_adapt(x)

        return outputs

    def forward_and_adapt(self, x):
        # Store mean and std of original network
        # extract stats from test image
        self.model.train()
        with torch.no_grad():
            intermediate_output = self.model(x)
        validation_mean, validation_variance = get_stats(self.model)
        self.model.eval()

        # interpolate between train and test stats
        interpolated_means, interpolated_stds = self.interpolate_statistics(self.original_mean, self.original_variance, validation_mean, validation_variance)

        # predict multiple logits
        predictions = []
        confidences = []

        for interpolated_mean, interpolated_std in zip(interpolated_means, interpolated_stds):
            self.model = update_stats(self.model, interpolated_mean, interpolated_std)
            prediction, confidence = make_prediction(self.model, x)

            predictions.append(prediction)
            confidences.append(confidence)

        # combine predictons
        outputs = self.weighted_average(predictions, confidences)
        # return outputs
        return outputs

    def weighted_average(self, preds, weights, normalize=False):
        
        preds = torch.concat(preds)
        weights = torch.concat(weights)

        if normalize:
                weights = (weights - weights.min()) / (weights.max() - weights.min())

        if weights is not None:
            # Normalize weights along the 0th dimension (the prediction axis) so they sum to 1 for each pixel
            weights = torch.softmax(weights/1.0, dim=0)  # Shape remains the same

            # Compute the weighted predictions
            weighted_pred = (preds * weights).sum(dim=0)  # Summing along the 0th dimension

        else:
            weighted_pred = preds.mean(0)
        mask_pred = weighted_pred.unsqueeze(0)
        
        return mask_pred
    
    def interpolate_statistics(self,original_mean, original_variance, validation_mean, validation_variance):
        interp_num = 5
        step_size = 1 / interp_num

        interpolated_means, interpolated_stds = [], []
        for i in range(interp_num+1):
            rate = i * step_size
            tmp_mean = [(1-rate)*m1.cpu()+rate*m2.cpu() for m1,m2 in zip(original_mean, validation_mean)]
            tmp_std =  [(1-rate)*s1.cpu()+rate*s2.cpu() for s1,s2 in zip(original_variance, validation_variance)]
            interpolated_means.append(tmp_mean)
            interpolated_stds.append(tmp_std)
        return interpolated_means, interpolated_stds