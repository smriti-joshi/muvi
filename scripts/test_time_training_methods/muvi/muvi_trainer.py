import torch
import torch.nn as nn
import time
from utils import segmentation_loss
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch.cuda.amp import GradScaler
from utils import check_sagittal_orientation, convert_plane, make_the_actual_prediction_with_permutation
from torch.autograd import Variable
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import yaml

class MuViTrainer:
    """
    Handles MuVi test-time training logic:
      - Multi-view pseudolabel generation
      - Consistency + feature alignment losses
      - Training loop per image
    """

    # -----------------------------
    # Initializing the trainer class
    # ----------------------------- 

    def __init__(self, model, configuration_manager, label_manager, slicer_fn, pred_fn, device):
        # super().__init__()
        self.model = model
        self.configuration_manager = configuration_manager
        self.label_manager = label_manager
        self.device = device
        self.scaler = GradScaler()
        self.slicer_fn = slicer_fn
        self.pred_fn = pred_fn


        with open("/workspace/muvi/scripts/test_time_training_methods/muvi/config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        self.epochs = self.cfg["training"]["epochs"]
        self.learning_rate = self.cfg["training"]["learning_rate"]
        self.optimizer = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.learning_rate,
                        weight_decay= self.cfg["training"]["optimizer"]["weight_decay"],
                        momentum= self.cfg["training"]["optimizer"]["momentum"],
                        nesterov= self.cfg["training"]["optimizer"]["nesterov"])
        self._initialize_loss()
      
    def _initialize_loss(self):
        self.losses = nn.ModuleDict({
            "segmentation": nn.ModuleDict({
                "axial": segmentation_loss(self.configuration_manager, self.label_manager),
                "sagittal": segmentation_loss(self.configuration_manager, self.label_manager),
                "coronal": segmentation_loss(self.configuration_manager, self.label_manager),
            }),
            "consistency": nn.ModuleDict({
                "c1": segmentation_loss(self.configuration_manager, self.label_manager),
                "c2": segmentation_loss(self.configuration_manager, self.label_manager),
            }),
            "feature": nn.ModuleDict({
                "f1": nn.CosineEmbeddingLoss(),
                "f2": nn.CosineEmbeddingLoss(),
                "f3": nn.CosineEmbeddingLoss(),
            })
        })

    # -----------------------------
    # MuVi Adaptation
    # ----------------------------- 
    
    def get_padded_data_and_slicers(self, input_image):

        data, _ = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                    'constant', {'value': 0}, True,
                                                    None)
        slicers = self.slicer_fn(data.shape[1:])
        return data, slicers
    
    def train_epoch(self, input_image, input_image_path):
        """
        Run one test-time training epoch with MuVi objectives.
        
        Args:
            input_image (torch.Tensor): Input volume (C, X, Y, Z).
            input_image_path (str): Path to the image (used for orientation checks).

        Returns:
            nn.Module: Updated model after one epoch of adaptation.
        """
        t1 = time.perf_counter()
        self.model.eval()
        scaler = GradScaler()
        train_outputs = []

        # -----------------------------
        # (1) Prepare padded slices & pseudo labels
        # -----------------------------
        is_sagittal = check_sagittal_orientation(input_image_path)
        data_axial, slicers_axial = (
            self.get_padded_data_and_slicers(
                convert_plane(input_image, 'one_flip_forward')
            ) if is_sagittal else 
            self.get_padded_data_and_slicers(input_image)
        )

        print('\n-------------- (1) Pseudolabel Computation ---------------')
        target_axial = self._get_pseudolabel(input_image, is_sagittal)

        print('\n-------------- (2) Test-time Training --------------------')

        # -----------------------------
        # (2) Loop over axial slices
        # -----------------------------
        with torch.enable_grad():
            for sl in slicers_axial:
                with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    
                    # Slice extraction
                    workon = Variable(data_axial[sl][None], requires_grad=True).to(self.device)
                    target = Variable(target_axial[sl][None], requires_grad=True).to(self.device)

                    # Prepare views in sagittal & coronal planes
                    workon_sag, target_sag, workon_cor, target_cor = self._get_reoriented_views(workon, target, is_sagittal)

                    # -----------------------------
                    # (3) Forward passes & losses
                    # -----------------------------
                    self.optimizer.zero_grad()

                    # Segmentation losses (axial, sagittal, coronal)
                    l_seg_1, output = self._compute_loss_and_output(self.losses["segmentation"]["axial"], workon, target)
                    l_seg_2, output_sag = self._compute_loss_and_output(self.losses["segmentation"]["sagittal"], workon_sag, target_sag)
                    l_seg_3, output_cor = self._compute_loss_and_output(self.losses["segmentation"]["coronal"], workon_cor, target_cor)

                    # Feature similarity losses
                    l_feat_1 = self._feature_loss(workon, workon_sag, self.losses["feature"]["f1"])
                    l_feat_2 = self._feature_loss(workon, workon_cor, self.losses["feature"]["f2"])

                    # Consistency losses
                    axial_pred = torch.unsqueeze(torch.argmax(output, dim=0), dim=0)
                    l_cons_1 = self.losses["consistency"]["c1"](output_sag.permute(0, 1, 3, 4, 2), axial_pred)
                    l_cons_2 = self.losses["consistency"]["c2"](output_cor.permute(0, 1, 4, 2, 3), axial_pred)

                    # -----------------------------
                    # (4) Backpropagation
                    # -----------------------------
                    total_loss = l_seg_1 + l_seg_2 + l_seg_3 + l_feat_1 + l_feat_2 + l_cons_1 + l_cons_2
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    scaler.step(self.optimizer)
                    scaler.update()

                    train_outputs.append(total_loss.detach().cpu().numpy())

            # -----------------------------
            # (5) Finish
            # -----------------------------
            t2 = time.perf_counter()
            print(f'Finished in {t2 - t1:.2f} seconds\n')
            print('----------------------------------\n')
        return self.model


    # -----------------------------
    # 🔹 Helper functions
    # -----------------------------
    def _get_reoriented_views(self, workon, target, is_sagittal):
        """Return sagittal & coronal reoriented views of inputs and targets."""
        if is_sagittal:
            workon_sag, target_sag = workon.permute(0, 1, 4, 2, 3), target.permute(0, 1, 4, 2, 3)
            workon_cor, target_cor = workon.permute(0, 1, 3, 4, 2), target.permute(0, 1, 3, 4, 2)
        else:
            workon_sag, target_sag = workon.permute(0, 1, 4, 2, 3), target.permute(0, 1, 4, 2, 3)
            workon_cor, target_cor = workon.permute(0, 1, 4, 2, 3).permute(0, 1, 4, 2, 3), target.permute(0, 1, 4, 2, 3).permute(0, 1, 4, 2, 3)
        return workon_sag, target_sag, workon_cor, target_cor


    def _compute_loss_and_output(self, loss_fn, inputs, targets):
        """Forward pass and loss computation."""
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        return loss, outputs


    def _feature_loss(self, output_a, output_b, loss_fn):
        """Compute feature similarity loss using encoder features."""        
        feat_a = torch.flatten(self.model.encoder(output_a)[-1])
        feat_b = torch.flatten(self.model.encoder(output_b)[-1])
        return loss_fn(
            torch.unsqueeze(feat_a, dim=1),
            torch.unsqueeze(feat_b, dim=1),
            torch.ones(feat_b.shape).to(self.device)
        )

    def _get_pseudolabel(self, input_image, is_it_sagittal: bool):
        """
        Compute pseudolabels in axial/sagittal/coronal planes with optional flips.
        """
        # define plane configs depending on orientation
        if is_it_sagittal:
            planes = [
                ("axial",     "one_flip",  self.cfg["pseudolabel"]["threshold_axial"]),
                ("sagittal",  None,        self.cfg["pseudolabel"]["threshold_sagittal"]),
                ("coronal",   "two_flips", self.cfg["pseudolabel"]["threshold_coronal"]),
            ]
        else:
            planes = [
                ("axial",     None,        self.cfg["pseudolabel"]["threshold_axial"]),
                ("sagittal",  "two_flips", self.cfg["pseudolabel"]["threshold_sagittal"]),
                ("coronal",   "one_flip",  self.cfg["pseudolabel"]["threshold_coronal"]),
            ]

        preds, threshed = [], []
        for i, (name, flip, thr) in enumerate(planes, 1):
            print(f"\n{name} ({i}/3)")
            pred = make_the_actual_prediction_with_permutation(input_image, flip, self.pred_fn)
            preds.append(pred)
            threshed.append(self._threshold_label(pred, threshold=thr))

        # combine all three thresholded predictions
        combined = torch.unsqueeze((sum(threshed) > 0).to(dtype=torch.float16), dim=0)

        # get padded data depending on orientation
        if is_it_sagittal:
            combined = convert_plane(combined, "one_flip_forward")

        target_axial, _ = self.get_padded_data_and_slicers(combined)
        return target_axial

       
    def _threshold_label(self, self_training_objective, threshold):
        entropy = self._calculate_entropy(self_training_objective)
        return torch.argmax(self_training_objective, axis =0) * entropy.le(threshold).to(dtype=torch.float16) 
    
    def _calculate_entropy(self, self_training_objective):
        prob = torch.softmax(self_training_objective, axis = 0)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=0)
        return entropy
    