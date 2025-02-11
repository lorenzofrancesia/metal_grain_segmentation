import numpy as np
import gudhi as gd
import torch
import torch.nn as nn
import math


class TopologicalLoss(nn.Module):
    """
    Topological Loss for Neural Network Segmentation.

    This class implements a topological loss function that encourages the network to
    produce segmentations with similar topological features to the ground truth.
    It leverages persistent homology to compare the topological structure of
    the predicted likelihood map with the ground truth.
    """

    def __init__(self, pers_thresh=0.03, pers_thresh_perfect=0.99, topo_size=100, device='cuda' if torch.cuda.is_available() else 'cpu', debugging=False):
        """
        Initializes the TopologicalLoss module.

        Args:
            pers_thresh (float, optional): Persistence threshold for filtering noise in diagrams. Defaults to 0.03.
            pers_thresh_perfect (float, optional): Threshold for considering a topological feature as a perfect match. Defaults to 0.99.
            topo_size (int, optional): Size of the patches used for topological analysis. Defaults to 100.
            device (str, optional): Device to run the loss calculation on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super(TopologicalLoss, self).__init__()
        self.pers_thresh = pers_thresh
        self.pers_thresh_perfect = pers_thresh_perfect
        self.topo_size = topo_size
        self.device = device  # Store the specified device
        self.debugging = debugging


    def forward(self, likelihood_tensor, gt_tensor):
        """
        Calculates the topology loss between the predicted likelihood and the ground truth.

        Args:
            likelihood_tensor (torch.Tensor): The likelihood output from the neural network (predicted segmentation). [B, N, H, W]
            gt_tensor (torch.Tensor): The ground truth segmentation. [B, 1, H, W]

        Returns:
            torch.Tensor: The calculated topological loss.
        """
        # Average over the different classes in the likelihood tensor
        loss = 0.0
        B = likelihood_tensor.shape[0]
        for b in range(B): #Iterate over batch
          loss += self.getTopoLoss(likelihood_tensor[b].mean(dim=0, keepdim=False), gt_tensor[b, 0, :, :]) # Average over N
        return loss/B # Normalize the loss


    def compute_dgm_force(self, lh_dgm, gt_dgm, do_return_perfect=False):
        """
        Computes the force to apply to the likelihood persistence diagram to match the ground truth.

        Args:
            lh_dgm (np.ndarray): Likelihood persistence diagram.
            gt_dgm (np.ndarray): Ground truth persistence diagram.
            do_return_perfect (bool, optional):  Whether to return perfect holes indices. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - force_list (np.ndarray): The force to apply to each point in the likelihood diagram.
                - idx_holes_to_fix (list): Indices of holes that need to be fixed (moved closer to perfect matches).
                - idx_holes_to_remove (list): Indices of holes that should be removed.
        """
        lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
        if gt_dgm.shape[0] == 0:
            gt_pers = None
            gt_n_holes = 0
        else:
            gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
            gt_n_holes = gt_pers.size  # number of holes in gt

        if gt_pers is None or gt_n_holes == 0:
            idx_holes_to_fix = []
            idx_holes_to_remove = list(set(range(lh_pers.size)))
            idx_holes_perfect = []
        else:
            # get "perfect holes" - holes which do not need to be fixed
            tmp = lh_pers > self.pers_thresh_perfect
            lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
            if np.sum(tmp) >= 1:
                lh_n_holes_perfect = tmp.sum()
                idx_holes_perfect = list(lh_pers_sorted_indices[:lh_n_holes_perfect])
            else:
                idx_holes_perfect = []

            # find top gt_n_holes indices
            idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]

            # the difference is holes to be fixed to perfect
            idx_holes_to_fix = list(
                set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect)
            )

            # remaining holes are all to be removed
            idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

        # only select the ones whose persistence is large enough
        pers_thd = self.pers_thresh
        idx_valid = np.where(lh_pers > pers_thd)[0]
        idx_holes_to_remove = list(
            set(idx_holes_to_remove).intersection(set(idx_valid))
        )

        force_list = np.zeros(lh_dgm.shape)

        # push each hole-to-fix to (0,1)
        force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
        force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

        # push each hole-to-remove to (0,1)
        force_list[idx_holes_to_remove, 0] = (
            lh_pers[idx_holes_to_remove] / math.sqrt(2.0)
        )
        force_list[idx_holes_to_remove, 1] = (
            -lh_pers[idx_holes_to_remove] / math.sqrt(2.0)
        )

        if do_return_perfect:
            return (
                force_list,
                idx_holes_to_fix,
                idx_holes_to_remove,
                idx_holes_perfect,
            )

        return force_list, idx_holes_to_fix, idx_holes_to_remove

    def getCriticalPoints(self, likelihood):
        """
        Computes the critical points of the image (Value range from 0 -> 1).

        Args:
            likelihood (torch.Tensor): Likelihood image from the output of the neural networks.

        Returns:
            tuple: A tuple containing:
                - pd_lh (np.ndarray): Persistence diagram.
                - bcp_lh (np.ndarray): Birth critical points.
                - dcp_lh (np.ndarray): Death critical points.
                - bool: Skip the process if number of matching pairs is zero.
        """
        lh = 1 - likelihood

        if lh.shape[0] == 0 or lh.shape[1] == 0:
            if self.debugging:
                print("WARNING: Encountered empty patch!")
            return None, None, None, False # Changed to None instead of 0 to avoid type issues later

        lh_np = lh.cpu().detach().numpy() # convert to numpy array for gudhi processing

        lh_vector = np.asarray(lh_np).flatten() # convert to numpy array

        lh_cubic = gd.CubicalComplex( # construct a gudhi cubical complex
            dimensions=[lh_np.shape[0], lh_np.shape[1]], # dimensions of cubical complex are hxw of likelihood
            top_dimensional_cells=lh_vector, # values of top dimensonal cells
        )

        Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0) # calculate the persistence diagram
        pairs_lh = lh_cubic.cofaces_of_persistence_pairs() # calculate persistence pairs

        # If the pairs is 0, return False to skip
        if len(pairs_lh[0]) == 0:
            return None, None, None, False

        # return persistence diagram, birth/death critical points
        pd_lh = np.array(
            [
                [lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]]
                for i in range(len(pairs_lh[0][0]))
            ]
        ) # persistence diagram
        bcp_lh = np.array(
            [
                [
                    pairs_lh[0][0][i][0] // lh_np.shape[1],
                    pairs_lh[0][0][i][0] % lh_np.shape[1],
                ]
                for i in range(len(pairs_lh[0][0]))
            ]
        ) # birth critical points
        dcp_lh = np.array(
            [
                [
                    pairs_lh[0][0][i][1] // lh_np.shape[1],
                    pairs_lh[0][0][i][1] % lh_np.shape[1],
                ]
                for i in range(len(pairs_lh[0][0]))
            ]
        ) # death critical points

        return pd_lh, bcp_lh, dcp_lh, True

    def getTopoLoss(self, likelihood, gt, topo_size=100):
        """
        Calculates the topology loss of the predicted image and ground truth image.
        Warning: To make sure the topology loss is able to back-propagation, likelihood
        tensor requires to clone before detach from GPUs. In the end, you can hook the
        likelihood tensor to GPUs device.

        Args:
            likelihood (torch.Tensor): The likelihood pytorch tensor. [H,W]
            gt (torch.Tensor): The groundtruth of pytorch tensor. [H,W]
            topo_size (int, optional): The size of the patch is used. Defaults to 100.

        Returns:
            torch.Tensor: The topology loss value (tensor).
        """

        topo_cp_weight_map = torch.zeros_like(likelihood, dtype=torch.float, device=self.device)
        topo_cp_ref_map = torch.zeros_like(gt, dtype=torch.float, device=self.device)
        loss_topo = torch.tensor(0.0, device=self.device, requires_grad=True)

        for y in range(0, likelihood.shape[0], topo_size):
            for x in range(0, likelihood.shape[1], topo_size):
                lh_patch = likelihood[
                    y : min(y + topo_size, likelihood.shape[0]),
                    x : min(x + topo_size, likelihood.shape[1]),
                ]
                gt_patch = gt[
                    y : min(y + topo_size, gt.shape[0]),
                    x : min(x + topo_size, gt.shape[1]),
                ]

                if torch.min(lh_patch) == 1 or torch.max(lh_patch) == 0:
                    if self.debugging:
                        print("Warning: lh_patch is uniform (all 0 or all 1). Skipping.")
                    continue
                if torch.min(gt_patch) == 1 or torch.max(gt_patch) == 0:
                    if self.debugging:
                      print("Warning: gt_patch is uniform (all 0 or all 1). Skipping.")
                    continue
                    
                try:
                    # Get the critical points of predictions and ground truth
                    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = self.getCriticalPoints(lh_patch)
                    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = self.getCriticalPoints(gt_patch)

                    # If the pairs not exist, continue for the next loop
                    if not (pairs_lh_pa):
                        if self.debugging:
                            print("Warning: No persistence pairs found in lh_patch. Skipping.")
                        continue
                    if not (pairs_lh_gt):
                        if self.debugging:
                            print("Warning: No persistence pairs found in gt_patch. Skipping.")
                        continue

                    if pd_lh is None or pd_gt is None:
                        if self.debugging:
                            print("Warning: pd_lh or pd_gt is None. Skipping.")
                        continue

                    (
                        force_list,
                        idx_holes_to_fix,
                        idx_holes_to_remove,
                    ) = self.compute_dgm_force(pd_lh, pd_gt) # removed pers_thresh

                    if len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0:
                        for hole_indx in idx_holes_to_fix:
                            if (
                                int(bcp_lh[hole_indx][0]) >= 0
                                and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]
                                and int(bcp_lh[hole_indx][1]) >= 0
                                and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]
                            ):
                                topo_cp_weight_map[
                                    y + int(bcp_lh[hole_indx][0]),
                                    x + int(bcp_lh[hole_indx][1]),
                                ] = 1  # push birth to 0 i.e. min birth prob or likelihood
                                topo_cp_ref_map[
                                    y + int(bcp_lh[hole_indx][0]),
                                    x + int(bcp_lh[hole_indx][1]),
                                ] = 0
                            if (
                                int(dcp_lh[hole_indx][0]) >= 0
                                and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]
                                and int(dcp_lh[hole_indx][1]) >= 0
                                and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]
                            ):
                                topo_cp_weight_map[
                                    y + int(dcp_lh[hole_indx][0]),
                                    x + int(dcp_lh[hole_indx][1]),
                                ] = 1  # push death to 1 i.e. max death prob or likelihood
                                topo_cp_ref_map[
                                    y + int(dcp_lh[hole_indx][0]),
                                    x + int(dcp_lh[hole_indx][1]),
                                ] = 1
                        for hole_indx in idx_holes_to_remove:
                            if (
                                int(bcp_lh[hole_indx][0]) >= 0
                                and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]
                                and int(bcp_lh[hole_indx][1]) >= 0
                                and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]
                            ):
                                topo_cp_weight_map[
                                    y + int(bcp_lh[hole_indx][0]),
                                    x + int(bcp_lh[hole_indx][1]),
                                ] = 1  # push birth to death  # push to diagonal
                                if (
                                    int(dcp_lh[hole_indx][0]) >= 0
                                    and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]
                                    and int(dcp_lh[hole_indx][1]) >= 0
                                    and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]
                                ):
                                    topo_cp_ref_map[
                                        y + int(bcp_lh[hole_indx][0]),
                                        x + int(bcp_lh[hole_indx][1]),
                                    ] = likelihood[
                                        int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])
                                    ]
                                else:
                                    topo_cp_ref_map[
                                        y + int(bcp_lh[hole_indx][0]),
                                        x + int(bcp_lh[hole_indx][1]),
                                    ] = 1
                            if (
                                int(dcp_lh[hole_indx][0]) >= 0
                                and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]
                                and int(dcp_lh[hole_indx][1]) >= 0
                                and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]
                            ):
                                topo_cp_weight_map[
                                    y + int(dcp_lh[hole_indx][0]),
                                    x + int(dcp_lh[hole_indx][1]),
                                ] = 1  # push death to birth # push to diagonal
                                if (
                                    int(bcp_lh[hole_indx][0]) >= 0
                                    and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]
                                    and int(bcp_lh[hole_indx][1]) >= 0
                                    and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]
                                ):
                                    topo_cp_ref_map[
                                        y + int(dcp_lh[hole_indx][0]),
                                        x + int(dcp_lh[hole_indx][1]),
                                    ] = likelihood[
                                        int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])
                                    ]
                                else:
                                    topo_cp_ref_map[
                                        y + int(dcp_lh[hole_indx][0]),
                                        x + int(dcp_lh[hole_indx][1]),
                                    ] = 0
                                    
                except Exception as e:
                    if self.debugging:
                        print(f"Exception occurred during topological loss calculation: {type(e).__name__}, {e}")
                        print(f"Skipping patch at y={y}, x={x}")
                    continue # Skip to the next patch

        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((likelihood * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
        return loss_topo