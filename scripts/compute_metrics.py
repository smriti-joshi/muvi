import os
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import csv
from scipy.spatial import cKDTree
import SimpleITK as sitk
import numpy as np
from medpy import metric

def hausdorff_distance_mask(image0, image1, method = 'standard'):
    # https://github.com/scikit-image/scikit-image/issues/6890
    """Calculate the Hausdorff distance between the contours of two segmentation masks.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the 
    contour of ``image0`` and its nearest point on the contour of ``image1``, and 
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((100, 100), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[30:71, 30:71] = disk(20)
    >>> predicted[25:65, 40:70] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    11.40175425099138
    """
    image0_array = sitk.GetArrayFromImage(sitk.LabelContour(sitk.ReadImage(image0)))
    image1_array = sitk.GetArrayFromImage(sitk.LabelContour(sitk.ReadImage(image1, sitk.sitkUInt8)))

    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    
    a_points = np.argwhere(image0_array>0)
    b_points = np.argwhere(image1_array>0)
    
    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))
    
  
def assd_eval(prediction_path,ground_truth_path,num_classes):
    #Average Symmetric Surface Distance (ASSD)
    predict = sitk.GetArrayFromImage(sitk.ReadImage(prediction_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(ground_truth_path))
    if len(np.unique(predict)) > 1 :
        assd_all = np.zeros(num_classes)
        for c in range(num_classes):
            reference = (label==c) * 1
            result = (predict==c) * 1
            assd_all[c] = metric.binary.assd(result,reference)
        return assd_all[1:]
    else:
        return np.array([150.0])
    
    
def compute_average_dice_and_iou(path_to_pred, path_to_gt, patient_ids, csv_path, text='dummy'):
   
    dice, haus, asd_list = [], [], []


    csv_array = []
    csv_array.append(['patient_id', 'dice', 'haus', 'assd'])

    for pred_id in patient_ids:
        if pred_id.endswith('.nii.gz'):
            gt_path = os.path.join(path_to_gt, pred_id)
            pred_path = os.path.join(path_to_pred, pred_id)
            patient_id = pred_id.split('.')[0]

            # Dice calculated from nnUNet
            results = compute_metrics(gt_path, pred_path, image_reader_writer=SimpleITKIO(), labels_or_regions= [0, 1])
            dice.append(results['metrics'][1]['Dice'])

            # Hausoff Distance
            haus_distance =  hausdorff_distance_mask(pred_path, gt_path, method='modified')
            if haus_distance is not np.inf:
                haus.append(haus_distance)
            else:
                haus.append(200)

            # Average Symmetric Surface Distance (ASSD)
            asd = assd_eval(pred_path, gt_path, num_classes=2)
            asd_list.append(asd[0])

            # Store results in csv
            csv_array.append([patient_id, results['metrics'][1]['Dice'], haus_distance, asd[0]])


    av_dice, std_dice = np.mean(dice), np.std(dice) 
    av_haus, std_haus = np.mean(haus), np.std(haus)
    av_asd, std_asd = np.mean(asd_list), np.std(asd_list)

    print('--------------------------------------------------------------------------------')
    print(f"Average dice: {av_dice:.4f} ({std_dice:.4f}),\n"
        f"Average Hausdorff: {av_haus:.4f} ({std_haus:.4f}),\n"
        f"Average ASD: {av_asd:.4f} ({std_asd:.4f})")
    print('--------------------------------------------------------------------------------')
     

    with open(os.path.join(csv_path, 'results.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(csv_array)
    
def main():   

    prediction_path = '/workspace/AutomaticSegmentation/reproducibility/ispy_muvi_instancenorm'
    # gt_path = '/data/automatic-segmentation-nnunet-raw/nnUNet_raw/Dataset102_DukeTCGAHalf/labelsTr'
    gt_path = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/data_external_validation/ISPY1/labelsTs_half'
    
    compute_average_dice_and_iou(prediction_path, gt_path, sorted(os.listdir(prediction_path)), csv_path=prediction_path)


if __name__ == "__main__":
    main()