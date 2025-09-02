import nibabel as nib
import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

def check_sagittal_orientation(nifti_file_path):
    # Load the NIfTI file
    img = nib.load(nifti_file_path)

    # Get the header
    header = img.header

    # Check the orientation information in the header
    # The 'dim_info' field and 'sform_code' can be useful
    dim_info = header['dim_info']
    sform_code = header['sform_code']

    # You may also want to inspect the 'srow_x', 'srow_y', and 'srow_z' fields
    srow_x = header['srow_x']
    srow_y = header['srow_y']
    srow_z = header['srow_z']

    # Check conditions based on the header information
    is_sagittal = (dim_info & 3 == 2) or (sform_code == 1 and srow_x[0] == 0)

    return is_sagittal


def segmentation_loss(configuration_manager, label_manager):
    return DC_and_CE_loss({'batch_dice': configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                    ignore_label=label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


def convert_plane(image, A2B):
    if A2B == 'one_flip_forward' :
        return image.permute(0, 2, 3, 1)
    elif A2B == 'one_flip_backward':
        return image.permute(0, 3, 1, 2)
    elif  A2B == 'two_flips_forward' :
        return image.permute(0, 2, 3, 1).permute(0, 2, 3, 1)
    elif A2B == 'two_flips_backward':
        return image.permute(0, 3, 1, 2).permute(0, 3, 1, 2)

def make_the_actual_prediction_with_permutation(input_image, flip_variable=None, pred_fn = None):
    if flip_variable==None:
        with torch.no_grad():
            prediction =  pred_fn(input_image)
    else:
        with torch.no_grad():
            prediction = convert_plane(pred_fn(convert_plane(input_image, flip_variable + '_forward')), flip_variable +'_backward')
    return prediction