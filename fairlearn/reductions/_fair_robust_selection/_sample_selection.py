import numpy as np
import itertools

import torch
import torch.nn as nn


class FairBatchSampler:
    """
    """

    def __init__(self, model, x, y, z, fairness_constraint, *,
                 loss_func=nn.BCELoss(), batch_size=128, warm_start=100,
                 tau=0.9, alpha=0.001):
        """
        Parameters
        ----------
        model : torch.nn.module
            A model implementing methods :code:`forward(x)`,
            where `x` is the pytorch tensor of features.
            predictions returned by  :code:`forward(x)`` are either 0 or 1.
        fairness_constraint : fairlearn.reductions.Moment
            The fairness constraints expressed as a :class:`~Moment`.
        loss_func : torch.nn.module.loss
            torch loss function
        batch_size : positive integetr
            the size of every batch
        tau: float
            number in range (0,1] indicating the clean ratio of the data.
        alpha : float
            A positive number for step size that used in the lambda adjustment.
        """

        self.model = model
        self.x_data = x
        self.y_data = y
        self.z_data = z
        self.fairness_constraint = fairness_constraint
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.tau = tau
        if self.tau > 1 or self.tau <= 0:
            raise ValueError("tau must be between (0,1]")
        self.alpha = alpha

        self.data_len = len(self.y_data)
        self.num_batches = int(self.data_len / self.batch_size)
        self.current_epoch = 0

        # Takes the unique values of the data
        self.z_values = list(self.z_data.unique())
        self.y_values = list(self.y_data.unique())
        self.yz_tuples = list(itertools.product(self.y_values, self.z_values))

        # Finds the index and len of each z and y value
        self.z_index = {}
        self.z_len = {}
        self.clean_z_index = {}
        self.clean_z_len = {}
        for z_value in self.z_values:
            self.z_index[z_value] = (self.z_data == z_value).nonzero()
            self.z_len[z_value] = len(self.z_index[z_value])
            self.clean_z_index[z_value] = self.z_index[z_value].copy()
            self.clean_z_len[z_value] = len(self.clean_z_index[z_value])

        self.y_index = {}
        self.y_len = {}
        self.clean_y_index = {}
        self.clean_y_len = {}
        for y_value in self.y_values:
            self.y_index[y_value] = (self.y_data == y_value).nonzero()
            self.y_len[y_value] = len(self.y_index[y_value])
            self.clean_y_index[y_value] = self.y_index[y_value].copy()
            self.clean_y_len[y_value] = len(self.clean_y_index[y_value])

        self.yz_index = {}
        self.yz_len = {}
        self.clean_yz_index = {}
        self.clean_yz_len = {}
        for yz_tuple in self.yz_tuples:
            self.yz_index[yz_tuple] = ((self.y_data == yz_tuple[0]) & (self.z_data == yz_tuple[1])).nonzero()
            self.yz_len[yz_tuple] = len(self.yz_index[yz_tuple])
            self.clean_yz_index[yz_tuple] = self.yz_index[yz_tuple].copy()
            self.clean_yz_len[yz_tuple] = len(self.clean_yz_index[yz_tuple])

        self.entire_index = np.arange(self.data_len)
        self.clean_index = self.entire_index.copy()

        # Default batch size for every (y,z) tuple
        self.S = {}
        for yz_tuple in self.yz_tuples:
            self.S[yz_tuple] = self.batch_size * (self.yz_len[yz_tuple] / self.data_len)

        # lambda values for fairness
        self.lb1 = self.S[1, 1] / (self.S[1, 1] + (self.S[1, 0]))
        self.lb2 = self.S[-1, 1] / (self.S[-1, 1] + (self.S[-1, 0]))

    def adjust_lambda(self):
        """
        Adjusts the lambda values in each pair (y,z)
        """

        y_pred = self.model.predict(self.x_data)

        if self.fairness_constraint == "DemographicParity":
            ones = np.ones(len(self.y_data))
            dp_loss = self.loss_func(y_pred.squeeze(), ones)

            yhat_yz = {}
            for tmp_yz in self.yz_tuples:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.clean_yz_index[tmp_yz]])) / self.clean_z_len[tmp_yz[1]]

            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])

            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha

            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1

            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1

    def select_fair_robust_sample(self):
        """
        Selects fair and robust samples and adjusts the lambda values for fairness.
        """

        y_pred = self.model.predict(self.x_data)

        self.adjust_lambda()

        loss = self.loss_func(y_pred, self.y_data)
        profit = torch.max(loss) - loss

        current_weight_sum = {}

        lb_ratio = {}

        for tmp_yz in self.yz_tuples:
            if tmp_yz == (1, 1):
                lb_ratio[tmp_yz] = self.lb1
            elif tmp_yz == (1, 0):
                lb_ratio[tmp_yz] = 1 - self.lb1
            elif tmp_yz == (-1, 1):
                lb_ratio[tmp_yz] = self.lb2
            elif tmp_yz == (-1, 0):
                lb_ratio[tmp_yz] = 1 - self.lb2

            current_weight_sum[tmp_yz] = 0

        # Greedy-based algorithm

        (_, sorted_index) = torch.topk(profit, len(profit), largest=True, sorted=True)

        clean_index = []

        total_selected = 0

        desired_size = int(self.tau * len(self.y_data))

        for j in sorted_index:
            tmp_y = self.y_data[j].item()
            tmp_z = self.z_data[j].item()
            current_weight_list = list(current_weight_sum.values())

            if total_selected >= desired_size:
                break
            if all(i < desired_size for i in current_weight_list):
                clean_index.append(j)

                current_weight_sum[(tmp_y, tmp_z)] += 2 - lb_ratio[(tmp_y, tmp_z)]
                current_weight_sum[(tmp_y, 1 - tmp_z)] += 1 - lb_ratio[(tmp_y, 1 - tmp_z)]
                current_weight_sum[(tmp_y * -1, tmp_z)] += 1
                current_weight_sum[(tmp_y * -1, 1 - tmp_z)] += 1

                total_selected += 1

        clean_index = torch.LongTensor(clean_index).cuda()

        # Update the variables
        self.clean_index = clean_index

        for tmp_z in self.z_values:
            combined = torch.cat((self.z_index[tmp_z], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]

            self.clean_z_index[tmp_z] = intersection

        for tmp_y in self.y_values:
            combined = torch.cat((self.y_index[tmp_y], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]

            self.clean_y_index[tmp_y] = intersection

        for tmp_yz in self.yz_tuples:
            combined = torch.cat((self.yz_index[tmp_yz], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]

            self.clean_yz_index[tmp_yz] = intersection

        for tmp_z in self.z_values:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])

        for tmp_y in self.y_values:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])

        for tmp_yz in self.yz_tuples:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])

        return clean_index

    def __iter__(self):
        pass

