import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets, ):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0).to("cpu")
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()

        # mask  
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_labels(self,cui_lists,device):
        #caculate labels
        num_logits = len(cui_lists)
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        for i in range(num_logits):
            search_keywords = cui_lists[i]
            for j in range(i,num_logits):
                match_keywords = cui_lists[j]
                if search_keywords == match_keywords and search_keywords[0] == 'C':
                    labels[i,j]=1
                    labels[j,i]=1
        labels = F.normalize(labels,dim=0)
        return labels
    
    def SoftCrossEntropy(self,inputs, target, reduction='average'):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        else:
            loss = torch.sum(torch.mul(log_likelihood, target))
        return loss

    def forward(self, image_features, text_features, logit_scale):
        device = text_features.device

        logits_per_image = logit_scale * image_features @ text_features.t()

        batch_size = image_features.shape[0]
        labels_image = torch.arange(batch_size).to(device)

        for i in range(batch_size):
            for j in range(i, batch_size):
                if torch.equal(image_features[i], image_features[j]) and i != j:
                    labels_image[j] = labels_image[i]

        total_loss = F.cross_entropy(logits_per_image, labels_image)
        
        return total_loss
