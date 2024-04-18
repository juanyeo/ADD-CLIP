import random
import torch
import torch.nn.functional as F

import logging
from torchvision.ops import roi_align
import numpy as np
from open_clip import get_cast_dtype
from .precision import get_autocast

class CLIPSelf:
    def __init__(self, args):
        '''Conceptual Caption embedding path'''
        embeddings = np.load("/shared/s2/lab01/jiwoosong/dataset/cc_clip_hand_craft_ViTB16.npy")

        cls_embeddings = embeddings
        cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
        cls_embeddings = cls_embeddings.to(args.device)
        cast_dtype = get_cast_dtype(args.precision)
        if cast_dtype is not None:
            cls_embeddings = cls_embeddings.to(dtype=cast_dtype)
        self.cls_embeddings = cls_embeddings

    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args, guide_model=None):
        if distributed:
            model = model.module
            dist_model = dist_model.module
            guide_model = guide_model.module

        images, normed_boxes, image_crops = batch       # note texts are not paired with images
        
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)
        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=False) # [30, 512]
            guide_dense_features = guide_model.get_intermediate_layers(images, n=1)[0][:,1:,:]
            guide_crop_features = guide_model(image_crops) # [30, 768]
        # [B, D, W, H]
        student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)

        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)
        normed_dense_features = F.normalize(student_dense_features, dim=1)
        normed_guide_features = F.normalize(guide_dense_features, dim=2)        
        normed_guide_crop_features = F.normalize(guide_crop_features, dim=-1)

        denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)

        '''Weighted loss for 1st loss'''
        guided_student_features = self.get_guided_student_boxes(denormed_boxes, dense_features=normed_dense_features, guide_crop=normed_guide_crop_features, guide_features=normed_guide_features)
        guided_student_features = F.normalize(guided_student_features, dim=-1)
        loss_cosine = 1.0 - (guided_student_features *
                        normed_teacher_features).sum(-1).mean()
        
        '''Guidance loss for 2nd loss'''
        loss_guide = self.loss_simmap_guidance(normed_dense_features, normed_guide_features)

        '''ZSCL loss for 3rd loss'''
        student_crop_features = model.encode_image(image_crops, normalize=False)
        normed_student_crop_features = F.normalize(student_crop_features, dim=-1)


        logit = model.logit_scale.exp()
        temperature = 2

        '''student_crop_features shape: [num of crops, dim(512)], cls_embeddings shape: [num of cls(133), dim(512)]'''
        student_cls_probs = ((normed_student_crop_features @ self.cls_embeddings.T) * logit / temperature)
        teacher_cls_probs = F.softmax((normed_teacher_features @ self.cls_embeddings.T) * logit / temperature, dim=-1)
        
        loss_distil = F.cross_entropy(student_cls_probs, teacher_cls_probs, reduction='mean') * (temperature**2)
        loss_distil *= 0.01

        losses = dict(loss_cosine=loss_cosine, loss_guide=loss_guide, loss_distil=loss_distil)

        return losses, len(images), model.logit_scale.exp()

    '''
    28 Nov 23: additional functions (START)
    '''
    def get_guided_student(self, dense_features, guide_features):
        n_input = dense_features.shape[0]
        guide_features_qk = torch.matmul(guide_features[:, :1, :], guide_features[:, 1:, :].transpose(1, 2)).squeeze(1)
        guide_similarity = F.softmax(guide_features_qk, dim=1)

        dense_features = dense_features.reshape(n_input, dense_features.shape[1], -1) # [2, 768, 4096]

        blist = []
        for i in range(n_input):
            blist.append(torch.matmul(guide_similarity[i], dense_features[i].transpose(0, 1)))
        return torch.stack(blist)

    # weighted average
    def get_guided_student_boxes(self, denormed_boxes, dense_features, guide_crop, guide_features):
        result = [] # dense_features: [2, 512, 64, 64] # guide_features: [2, 4097, 768]
        box_index = 0

        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]
                cropped_student = dense_features[i, :, box[1]:box[3], box[0]:box[2]] # [512, 64, 22]
                cropped_guide = guide_features[i].reshape(dense_features.shape[2], dense_features.shape[3], guide_features.shape[2])[box[1]:box[3], box[0]:box[2], :] 
                cropped_guide = cropped_guide.reshape(-1, guide_features.shape[2]) # [1408, 768]
            
                guide_features_qk = torch.matmul(guide_crop[box_index], cropped_guide.transpose(0, 1)) # [1, 1408]
                guide_similarity = F.softmax(guide_features_qk, dim=0).detach()

                cropped_student = cropped_student.reshape(cropped_student.shape[0], -1)
                weighted_align = torch.matmul(guide_similarity, cropped_student.transpose(0, 1)) # [1, 512]

                result.append(weighted_align)
                box_index += 1

        return torch.stack(result)
    
    def loss_simmap_guidance(self, dense_features, guide_features):
        result = 0
        n_input = dense_features.shape[0]
        
        dense_features = dense_features.reshape(n_input, dense_features.shape[1], -1) # [2, 768, 4096]
        guide_features = guide_features.transpose(1, 2) # [2, 512, 4096]
        for i in range(n_input):
            student_sim_map = torch.matmul(dense_features[i].T, dense_features[i])
            guide_sim_map = torch.matmul(guide_features[i].T, guide_features[i])
            result += F.mse_loss(student_sim_map, guide_sim_map)

        return result / n_input

    def _denormalize_boxes(self, normed_boxes, x):
        if isinstance(x, tuple):
            h, w = x
        else:
            h, w = x.shape[-2:]
        
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    '''
    28 Nov 23: additional functions (END)
    '''


