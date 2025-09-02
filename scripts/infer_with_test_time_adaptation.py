import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join

import os
from typing import Union
import numpy as np
import glob

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from utils import make_the_actual_prediction_with_permutation

from test_time_training_methods.muvi.muvi_trainer import MuViTrainer
from test_time_training_methods.tent import tent
from test_time_training_methods.tent import norm
from test_time_training_methods.bnadapt import bn
from test_time_training_methods.intent import intent
from test_time_training_methods.memo import memo

class nnUnetTestTimeTrainer(nnUNetPredictor):

    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 allow_test_time_da: bool = True,
                 method = 'muvi') -> None:
        
       
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.version.cuda)  # Should print the CUDA version PyTorch is using
        print(torch.backends.cudnn.version())
        print(torch.cuda.get_device_name(0))

        super().__init__(tile_step_size,
                    use_gaussian,
                    use_mirroring,
                    perform_everything_on_gpu,
                    device,
                    verbose,
                    verbose_preprocessing,
                    allow_tqdm)

        self.method = method

    # -----------------------------
    # 🔹 Overloaded functions
    # -----------------------------
    def initialize_from_trained_model_folder(self, model_training_output_dir, use_folds, checkpoint_name: str = 'checkpoint_final.pth'):
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
        self.test_time_trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    self.trainer_name, 'nnunetv2.training.nnUNetTrainer')
        self.num_segmentation_heads =  self.plans_manager.get_label_manager(self.dataset_json).num_segmentation_heads
   
    
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)  
        empty_cache(self.device)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'
                # which method
                self.test_time_train(input_image)  

                if self.method != 'muvi':              
                    prediction = super().predict_sliding_window_return_logits(input_image)
                else:
                    prediction1 = make_the_actual_prediction_with_permutation(input_image, None, self.parent_pred_fn)  
                    prediction2 = make_the_actual_prediction_with_permutation(input_image, 'one_flip', self.parent_pred_fn)              
                    prediction3 = make_the_actual_prediction_with_permutation(input_image, 'two_flips', self.parent_pred_fn)  
                    prediction = (prediction1 + prediction2 + prediction3)/3
                self.case_count += 1

        return prediction

    def predict_from_files(self,
                           list_of_lists_or_source_folder,
                           output_folder_or_list_of_truncated_output_files,
                           save_probabilities,
                           overwrite,
                           num_processes_preprocessing,
                           num_processes_segmentation_export,
                           folder_with_segs_from_prev_stage,
                           num_parts,
                           part_id):
        
        self.source_files = glob.glob(os.path.join(list_of_lists_or_source_folder, '*.nii.gz'))

        # Count the number of files found
        self.case_count = 0

        super().predict_from_files(list_of_lists_or_source_folder,
                           output_folder_or_list_of_truncated_output_files,
                           save_probabilities,
                           overwrite,
                           num_processes_preprocessing,
                           num_processes_segmentation_export,
                           folder_with_segs_from_prev_stage,
                           num_parts,
                           part_id)
      
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None

        prediction = self.test_time_network(x)
       
        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

            num_predictons = 2 ** len(mirror_axes)
            if 0 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (2,))), (2,))
            if 1 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (3,))), (3,))
            if 2 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (4,))), (4,))
            if 0 in mirror_axes and 1 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (2, 3))), (2, 3))
            if 0 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (2, 4))), (2, 4))
            if 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (3, 4))), (3, 4))
            if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.test_time_network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
            prediction /= num_predictons
        return prediction
    
    def initialize_source_model(self):
        # initialize test time network and load pretrained weights
        self.test_time_network = self.test_time_trainer_class.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        self.test_time_network.load_state_dict(self.list_of_parameters[0])

        
        if ('nnUNet_compile' in os.environ.keys()) and (
                os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
            self.test_time_network = torch.compile(self.test_time_network)

    # -----------------------------
    # 🔹 Test time training
    # -----------------------------
    def test_time_train(self, input_image):

        self.initialize_source_model()

        if self.method == 'muvi':
            self.test_time_training_muvi(input_image)

        elif self.method == 'tent':
            self.test_time_training_tent()

        elif self.method == 'ptn':
            self.test_time_training_norm()

        elif self.method == 'bnadapt':
            self.test_time_training_bnadapt()

        elif self.method == 'intent':
            self.test_time_training_intent()
        
        elif self.method == 'memo':
            self.test_time_training_memo()
        else:
            print('this method does not exist or is not implemented')

    def test_time_training_muvi(self, input_image):

        self.parent_pred_fn = lambda x: super(nnUnetTestTimeTrainer, self).predict_sliding_window_return_logits(x)
            
        self.muvi_trainer = MuViTrainer(
            model=self.test_time_network,
            configuration_manager=self.configuration_manager,
            label_manager=self.label_manager,
            slicer_fn=self._internal_get_sliding_window_slicers,
            pred_fn=self.parent_pred_fn,
            device=self.device
        )
        self.test_time_network = self.muvi_trainer.train_epoch(input_image, self.source_files[self.case_count])

    def test_time_training_tent(self):        
        """Set up test-time normalization adaptation - Tent

        Adapts by training the affine parameters of batch norm (bn) 
        layers and using test statistics.
        """

        Tent = tent.configure_model(self.test_time_network)
        params, param_names = tent.collect_params(Tent)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        tented_model = tent.Tent(Tent, optimizer)
        self.test_time_network = tented_model
   

    def test_time_training_norm(self):
    
        """Set up test-time normalization adaptation - PTN

        Adapts by normalizing features with test statistics.
        no running average or other cross-batch estimation is used.
        """
        self.test_time_network = norm.Norm(self.test_time_network, no_stats=True)
     
    
    def test_time_training_bnadapt(self):
        """Set up test-time normalization adaptation - BNAdapt
        
        Adapts by adaptating batch-norm running mean through a combination of source
        and target features weighted by parameter N. 
        In our case, N = 16, n = 1. Therefore, prior = 0.94"""

        bnadapted = bn.adapt_bayesian(self.test_time_network, prior= 0.94)
        self.test_time_network = bnadapted

    def test_time_training_intent(self):

        """Set up test-time normalization adaptation - InTent
        
        Adapts by weighted ensembling of the predictions obtained from interpolating
        batch norm stats between source and targer distribution"""

        self.test_time_network = intent.InTent(model=self.test_time_network)

    def test_time_training_memo(self):
        """Set up test-time normalization adaptation - Memo
        
        Memo consistency between augmentations and minimizing entropy"""

        self.test_time_network = memo.Memo(model=self.test_time_network, prior=16)

if __name__ == '__main__':
    # predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    predictor = nnUnetTestTimeTrainer(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
        allow_test_time_da=True,
        method='muvi'
        )
    
    predictor.initialize_from_trained_model_folder(
    os.path.join(nnUNet_results, 'Dataset101_DukePhaseOneHalfMultifocal/nnUNetTrainer__nnUNetPlans__3d_fullres'),
    use_folds=[0],
    checkpoint_name='checkpoint_best.pth')
     
    predictor.predict_from_files(os.path.join(nnUNet_raw, '/data/automatic-segmentation-nnunet-raw/nnUNet_raw/Dataset105_ISPY1Half/imagesTs'),
                                os.path.join(nnUNet_raw, '/workspace/AutomaticSegmentation/reproducibility/ISPY_muvi_instancenorm_REFACTORED'),
                                save_probabilities=False, overwrite=False,
                                num_processes_preprocessing=1, num_processes_segmentation_export=1, folder_with_segs_from_prev_stage = None, num_parts=1, part_id=0)


