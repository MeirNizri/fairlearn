import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FairSampler:
    """
    This class implementing the lambda adjustment and batch selection
    of FairBatch [Roh et al., ICLR 2021] with robust training.
    """

    def __init__(self, model, x, y, z, fairness_constraint, *,
                 batch_size=128, warm_start=100, tau=0.9, alpha=0.001):
        """
        Parameters
        ----------
        model : torch.nn.module
            A model implementing methods :code:`forward(x)`,
            where `x` is the pytorch tensor of features.
            predictions returned by  :code:`forward(x)`` are either 0 or 1.
        fairness_constraint : fairlearn.reductions.Moment
            The fairness constraints expressed as a :class:`~Moment`.
        batch_size : positive integer
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
        self.loss_func = nn.BCELoss(reduction='none')
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
        self.z_values = self.z_data.unique().tolist()
        self.y_values = self.y_data.unique().tolist()
        self.yz_tuples = list(itertools.product(self.y_values, self.z_values))

        # Finds the index and len of each z and y value
        self.z_index = {}
        self.z_len = {}
        self.clean_z_index = {}
        self.clean_z_len = {}
        for z_value in self.z_values:
            self.z_index[z_value] = (self.z_data == z_value).nonzero().squeeze()
            self.z_len[z_value] = len(self.z_index[z_value])
            self.clean_z_index[z_value] = self.z_index[z_value].clone()
            self.clean_z_len[z_value] = len(self.clean_z_index[z_value])

        self.y_index = {}
        self.y_len = {}
        self.clean_y_index = {}
        self.clean_y_len = {}
        for y_value in self.y_values:
            self.y_index[y_value] = (self.y_data == y_value).nonzero().squeeze()
            self.y_len[y_value] = len(self.y_index[y_value])
            self.clean_y_index[y_value] = self.y_index[y_value].clone()
            self.clean_y_len[y_value] = len(self.clean_y_index[y_value])

        self.yz_index = {}
        self.yz_len = {}
        self.clean_yz_index = {}
        self.clean_yz_len = {}
        for yz_tuple in self.yz_tuples:
            self.yz_index[yz_tuple] = ((self.y_data == yz_tuple[0]) & (self.z_data == yz_tuple[1])).nonzero().squeeze()
            self.yz_len[yz_tuple] = len(self.yz_index[yz_tuple])
            self.clean_yz_index[yz_tuple] = self.yz_index[yz_tuple].clone()
            self.clean_yz_len[yz_tuple] = len(self.clean_yz_index[yz_tuple])

        self.clean_index = []

        # Default batch size for every (y,z) tuple
        self.S = {}
        for yz_tuple in self.yz_tuples:
            self.S[yz_tuple] = self.batch_size * (self.yz_len[yz_tuple] / self.data_len)

        # lambda values for fairness
        self.lb1 = self.S[(1, 1)] / (self.S[(1, 1)] + self.S[(1, 0)])
        self.lb2 = self.S[(0, 1)] / (self.S[(0, 1)] + self.S[(0, 0)])

    def adjust_lambda(self, y_pred):
        """
        Adjusts the lambda values in each pair (y,z)
        """
        if self.fairness_constraint.short_name == "DemographicParity":
            ones = torch.FloatTensor(np.ones(self.data_len)).squeeze()
            loss = self.loss_func(y_pred, ones)

            # get sum of loss in relation to each sensitive group
            yhat_yz = {}
            for yz_tuple in self.yz_tuples:
                yhat_yz[yz_tuple] = float(torch.sum(loss[self.clean_yz_index[yz_tuple]])) / self.clean_z_len[
                    yz_tuple[1]]

            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])

            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha

        self.lb1 = min(self.lb1, 1)
        self.lb1 = max(0, self.lb1)
        self.lb2 = min(self.lb2, 1)
        self.lb2 = max(0, self.lb2)

    def select_fair_robust_sample(self):
        """
        Selects fair and robust samples and adjusts the lambda values for fairness.
        """

        # get current evaluation and adjust lambda values
        self.model.eval()
        y_pred = self.model(self.x_data)
        self.adjust_lambda(y_pred)

        # Greedy-based algorithm
        loss = self.loss_func(y_pred, self.y_data)
        profit = torch.max(loss) - loss
        lambda_ratio = {(1, 1): self.lb1, (1, 0): 1 - self.lb1, (0, 1): self.lb2, (0, 0): 1 - self.lb2}

        self.clean_index = []
        total_selected = 0
        desired_size = int(self.tau * self.data_len)
        sum_selected_yz = {}
        for yz_tuple in self.yz_tuples:
            sum_selected_yz[yz_tuple] = 0

        # sort all items by their loss
        _, sorted_index = torch.sort(profit, descending=True)
        for idx in sorted_index:
            y_value = self.y_data[idx].item()
            z_value = self.z_data[idx].item()
            current_weight_list = list(sum_selected_yz.values())

            if total_selected >= desired_size:
                break
            elif all(i < desired_size for i in current_weight_list):
                self.clean_index.append(idx)

                sum_selected_yz[(y_value, z_value)] += 2 - lambda_ratio[(y_value, z_value)]
                sum_selected_yz[(y_value, 1 - z_value)] += 1 - lambda_ratio[(y_value, 1 - z_value)]
                sum_selected_yz[(1 - y_value, z_value)] += 1
                sum_selected_yz[(1 - y_value, 1 - z_value)] += 1

                total_selected += 1

        self.clean_index = torch.LongTensor(self.clean_index)

        for z_value in self.z_values:
            combined = torch.cat((self.z_index[z_value], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            self.clean_z_index[z_value] = intersection

        for y_value in self.y_values:
            combined = torch.cat((self.y_index[y_value], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            self.clean_y_index[y_value] = intersection

        for yz_tuple in self.yz_tuples:
            combined = torch.cat((self.yz_index[yz_tuple], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            self.clean_yz_index[yz_tuple] = intersection

        # update lengths
        for z_value in self.z_values:
            self.clean_z_len[z_value] = len(self.clean_z_index[z_value])
        for y_value in self.y_values:
            self.clean_y_len[y_value] = len(self.clean_y_index[y_value])
        for yz_tuple in self.yz_tuples:
            self.clean_yz_len[yz_tuple] = len(self.clean_yz_index[yz_tuple])

    def select_batch(self, batch_size, full_index):
        """
        Selects a certain number of batches based on the given batch size.

        Args:
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
        Returns:
            Indexes that indicate the data.
        """

        select_index = []
        for _ in range(self.num_batches):
            select_index.append(np.random.choice(full_index, batch_size, replace=False))
        return select_index

    def decide_fair_batch_size(self):
        """
        Calculates each class size based on the lambda values.
        Returns:
            Each class size for fairness.
        """

        for yz_tuple in self.yz_tuples:
            self.S[yz_tuple] = self.batch_size * self.clean_yz_len[yz_tuple] / len(self.clean_index)

        yz_sizes = {(1, 1): round(self.lb1 * (self.S[(1, 1)] + self.S[(1, 0)])),
                    (1, 0): round((1 - self.lb1) * (self.S[(1, 1)] + self.S[(1, 0)])),
                    (0, 1): round(self.lb2 * (self.S[(0, 1)] + self.S[(0, 0)])),
                    (0, 0): round((1 - self.lb2) * (self.S[(0, 1)] + self.S[(0, 0)]))}
        return yz_sizes

    def __iter__(self):
        """
        Iters the full process of fair and robust sample selection for serving the batches to training.
        Returns:
            Indexes that indicate the data in each batch.
        """
        self.current_epoch += 1

        if self.current_epoch > self.warm_start:
            self.select_fair_robust_sample()
            yz_sizes = self.decide_fair_batch_size()

            # Get the indices for each class
            sort_index_y_1_z_1 = self.select_batch(yz_sizes[(1, 1)], self.clean_yz_index[(1, 1)])
            sort_index_y_0_z_1 = self.select_batch(yz_sizes[(0, 1)], self.clean_yz_index[(0, 1)])
            sort_index_y_1_z_0 = self.select_batch(yz_sizes[(1, 0)], self.clean_yz_index[(1, 0)])
            sort_index_y_0_z_0 = self.select_batch(yz_sizes[(0, 0)], self.clean_yz_index[(0, 0)])

            for i in range(self.num_batches):
                key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))
                np.random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch

        else:
            entire_index = torch.arange(0, self.data_len)
            sort_index = self.select_batch(self.batch_size, entire_index)

            for i in range(self.num_batches):
                yield sort_index[i]


class CustomDataset(Dataset):
    """
    Attributes:
        x: A PyTorch tensor for features of the data.
        y: A PyTorch tensor for true labels of the data.
        z: A PyTorch tensor for sensitive feature of the data.
    """

    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors"""
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor

    def __getitem__(self, index):
        """Returns the selected data based on the index information"""
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        """Returns the length of data"""
        return len(self.x)
