import torch
import numpy as np

class Translate(object):

    def __call__(self, sample):
        in_fields, tgt_fields = sample

        new_in_fields = []
        new_tgt_fields = []

        for in_img, tgt_img in zip(in_fields, tgt_fields):

            axis = torch.randint(3, (1,)).item()

            # Does it need to be half the pixels?
            x_shift = torch.randint(in_img.shape[0], (1,)).item()
            y_shift = torch.randint(in_img.shape[1], (1,)).item()

            if axis == 0:
                x_shift = 0
            elif axis == 1:
                y_shift = 0
            else:
                axis = (1, 0)

            in_img = np.roll(in_img, (x_shift, y_shift), axis=axis)
            tgt_img = np.roll(tgt_img, (x_shift, y_shift), axis=axis)

            new_in_fields.append(in_img)
            new_tgt_fields.append(tgt_img)

        in_fields, tgt_fields = torch.tensor(new_in_fields), torch.tensor(new_tgt_fields)

        return in_fields, tgt_fields