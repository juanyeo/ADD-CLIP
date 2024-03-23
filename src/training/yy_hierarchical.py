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

        mask = torch.zeros(5, 21)
        mask[0][1:5] = 0.125 # (0.5 / 4)
        mask[1][[5, 6, 9, 10]] = 0.03125 # (0.5 / 4 / 4)
        mask[2][[7, 8, 11, 12]] = 0.03125
        mask[3][[13, 14, 17, 18]] = 0.03125
        mask[4][[15, 16, 19, 20]] = 0.03125
        mask = mask.to(device=device, dtype=cast_dtype, non_blocking=True)
        
        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops[:5], normalize=False)
        student_crop_features = model.encode_image(image_crops, normalize=False)

        normed_student_features = F.normalize(student_crop_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

        loss_hierarchical = self.loss_hierarchical(normed_teacher_features, normed_student_features, mask)
        losses = dict(loss_hierarchical=loss_hierarchical)

        return losses, len(images), model.logit_scale.exp()

    def loss_hierarchical(self, teacher_features, student_features, mask):
        result = 0
        n_input = teacher_features.shape[0] // 5
        
        for i in range(n_input):
            teacher = teacher_features[i*5:i*5+5] # 5, 512
            student = student_features[i*21:i*21+21] # 21, 512
            cosine_map = 1 - torch.matmul(teacher, student.T)
            result += (cosine_map * mask).sum()
            
        return result / n_input

    # AssertionError: No inf checks were recorded for this optimizer.
    def loss_hierarchical_sep(self, teacher_features, student_features):
        result = 0
        n_input = teacher_features.shape[0] // 21
        
        level2_indices = [5, 7, 13, 15]
        for i in range(n_input):
            level1_teacher = teacher_features[i*21]
            level1_student = student_features[(i*21)+1:(i*21)+5]
            cosine_map = 1 - torch.matmul(level1_teacher, level1_student.T)
            level1_result = cosine_map.mean()

            level2_result = 0
            for j in range(len(level2_indices)):
                start = level2_indices[j]
                level2_teacher = teacher_features[i*21+j+1]
                level2_student = student_features[[(i*21)+start,(i*21)+start+1,(i*21)+start+4,(i*21)+start+5]]
                cosine_map = 1 - torch.matmul(level2_teacher, level2_student.T)
                level2_result += cosine_map.mean()
            level2_result /= 4
            result += (level1_result + level2_result)/2
            
        return result / n_input

    # 0 [1, 2, 3, 4], 
    # 1 [5, 6, 9, 10], 2 [7, 8, 11, 12] 3 [13, 14, 17, 18] 4 [15, 16, 19, 20]

