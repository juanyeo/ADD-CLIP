import random
import torch
import torch.nn.functional as F

import logging
from torchvision.ops import roi_align

class CLIPSelf:
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
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
        student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False, extract_type=args.extract_type)

        
        '''
        28 Nov 23: extract dense features (START)
        ''' # [B, D, W, H]
        student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)
        with torch.no_grad(): # [B, HW, D]
            guide_dense_features = guide_model.get_intermediate_layers(images, n=12)[11][:,1:,:]
        # denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)
        # student_roi_features = roi_align(student_dense_features, denormed_boxes, (1, 1), 1.0, -1, True)[..., 0, 0]
        # if normalize: roi_features = F.normalize(roi_features, dim=-1)
        '''
        28 Nov 23: extract dense features (END)
        '''
        
        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)
        
        loss_cosine = 1.0 - (normed_student_features *
                             normed_teacher_features).sum(-1).mean()

        '''
        28 Nov 23: inter crop losses (START)
        '''
        # Normalize
        # normed_dense_features = F.normalize(student_dense_features, dim=1)
        # normed_guide_features = F.normalize(guide_dense_features, dim=2)

        loss_guide = self.loss_simmap_guidance_mse(student_dense_features, guide_dense_features)
        losses = dict(loss_cosine=loss_cosine*args.cosine_weight, loss_guide=loss_guide*1000)
        # loss_inter = self.loss_inter_features(denormed_boxes, normed_dense_features)
        #loss_inter = self.loss_inter_features_weighted(denormed_boxes, student_dense_features)
        # losses = dict(loss_cosine=loss_cosine*args.cosine_weight, loss_inter=loss_inter)
        
        '''
        28 Nov 23: inter crop losses (END)
        '''

        return losses, len(images), model.logit_scale.exp()

    '''
    28 Nov 23: additional functions (START)
    '''
    def loss_simmap_guidance_mse(self, dense_features, guide_features):
        result = 0
        n_input = dense_features.shape[0]
        
        dense_features = dense_features.reshape(n_input, dense_features.shape[1], -1) # [2, 768, 4096]
        guide_features = guide_features.transpose(1, 2) # [2, 512, 4096]
        for i in range(n_input):
            student_sim_map = torch.matmul(dense_features[i].T, dense_features[i])
            guide_sim_map = torch.matmul(guide_features[i].T, guide_features[i])
            
            # Softmax
            student_sim_map = F.softmax(student_sim_map/0.1, dim=1)
            guide_sim_map = F.softmax(guide_sim_map/0.1, dim=1)
            
            result += F.mse_loss(student_sim_map, guide_sim_map)

        return result / n_input

    def loss_simmap_guidance_ce(self, dense_features, guide_features):
        result = 0
        n_input = dense_features.shape[0]
        
        dense_features = dense_features.reshape(n_input, dense_features.shape[1], -1) # [2, 768, 4096]
        guide_features = guide_features.transpose(1, 2) # [2, 512, 4096]
        for i in range(n_input):
            student_sim_map = torch.matmul(dense_features[i].T, dense_features[i])
            guide_sim_map = torch.matmul(guide_features[i].T, guide_features[i])
            clip_sim_softmax = F.softmax(student_sim_map, dim=1)
            dino_sim_softmax = F.softmax(guide_sim_map, dim=1)
            
            result += torch.mean(-torch.sum(dino_sim_softmax * torch.log(clip_sim_softmax), 1))
            # result += -(dino_sim_softmax * (clip_sim_softmax).log()).sum(dim=1).mean(dim=0)
            # result -= (dino_sim_softmax * (clip_sim_softmax + 1e-7).log()).sum()
            # result += (dino_sim_softmax * (dino_sim_softmax / clip_sim_softmax).log()).sum()
            
        return result / n_input

    def loss_simmap_guidance_kl(self, dense_features, guide_features):
        result = 0
        n_input = dense_features.shape[0]
        
        dense_features = dense_features.reshape(n_input, dense_features.shape[1], -1) # [2, 768, 4096]
        guide_features = guide_features.transpose(1, 2) # [2, 512, 4096]
        for i in range(n_input):
            student_sim_map = torch.matmul(dense_features[i].T, dense_features[i])
            guide_sim_map = torch.matmul(guide_features[i].T, guide_features[i])
            clip_sim_softmax = F.softmax(student_sim_map, dim=1)
            dino_sim_softmax = F.softmax(guide_sim_map, dim=1)

            result += (dino_sim_softmax * (dino_sim_softmax / clip_sim_softmax).log()).sum(dim=1).mean(dim=0)

        return result / n_input

    def _denormalize_boxes(self, normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    def loss_inter_features(self, denormed_boxes, dense_features):
        result = 0
        count = 0
        
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]
                
                cropped_features = dense_features[i, :, box[0]:box[2], box[1]:box[3]]
                #avg_features = cropped_features.reshape(cropped_features.shape[0], -1).mean(1)
                cropped_features = cropped_features.reshape(cropped_features.shape[0], -1).transpose(0, 1)
                dot_matrix = torch.matmul(cropped_features, cropped_features.T).fill_diagonal_(0) #.sum() / 2
                result += torch.mean(dot_matrix)
                count += 1

        return result / count

    def loss_inter_features_weighted(self, denormed_boxes, dense_features):
        result = 0
        count = 0
        
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]
                
                cropped_features = dense_features[i, :, box[0]:box[2], box[1]:box[3]]
                cropped_features = cropped_features.reshape(cropped_features.shape[0], -1).transpose(0, 1)
                
                dot_matrix = torch.matmul(cropped_features, cropped_features.T)
                weighted_dot_matrix = F.softmax(dot_matrix.clone().fill_diagonal_(-float("Inf")), dim=1) * dot_matrix
                result += torch.sum(weighted_dot_matrix) / cropped_features.shape[0]
                count += 1

        return result / count
    
    '''
    28 Nov 23: additional functions (END)
    '''


