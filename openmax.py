import numpy as np
import scipy.spatial.distance as spd
import torch

import libmr

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import utils

import os.path as osp
import torch.distributed as dist


class Evaluation(object):
    """Evaluation class based on python list"""
    def __init__(self, predict, label,prediction_scores = None):
        self.predict = predict
        self.label = label
        self.prediction_scores = prediction_scores

        self.accuracy = self._accuracy()
        self.f1_measure = self._f1_measure()
        self.f1_macro = self._f1_macro()
        self.f1_macro_weighted = self._f1_macro_weighted()
        self.precision, self.recall = self._precision_recall(average='micro')
        self.precision_macro, self.recall_macro = self._precision_recall(average='macro')
        self.precision_weighted, self.recall_weighted = self._precision_recall(average='weighted')
        self.confusion_matrix = self._confusion_matrix()
        # if self.prediction_scores is not None:
        #     self.area_under_roc = self._area_under_roc(prediction_scores)

    def _accuracy(self) -> float:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _f1_measure(self) -> float:
        """
        Returns the F1-measure with a micro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='micro')

    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')

    def _f1_macro_weighted(self) -> float:
        """
        Returns the F1-measure with a weighted macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='weighted')

    def _precision_recall(self, average) -> (float, float):
        """
        Returns the precision and recall scores for the label and predictions. Observes the average type.

        :param average: string, [None (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
            For explanations of each type of average see the documentation for
            `sklearn.metrics.precision_recall_fscore_support`
        :return: float, float: representing the precision and recall scores respectively
        """
        assert len(self.predict) == len(self.label)
        precision, recall, _, _ = precision_recall_fscore_support(self.label, self.predict, average=average)
        return precision, recall

    def _area_under_roc(self, prediction_scores: np.array = None, multi_class='ovo') -> float:
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)

    def _confusion_matrix(self, normalize=None) -> np.array:
        """
        Returns the confusion matrix corresponding to the labels and predictions.

        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :return:
        """
        assert len(self.predict) == len(self.label)
        return confusion_matrix(self.label, self.predict, normalize=normalize)

    def plot_confusion_matrix(self, labels: [str] = None, normalize=None, ax=None, savepath=None) -> None:
        """

        :param labels: [str]: label names
        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :param ax: matplotlib.pyplot axes to draw the confusion matrix on. Will generate new figure/axes if None.
        :return:
        """
        conf_matrix = self._confusion_matrix(normalize)  # Evaluate the confusion matrix
        display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)  # Generate the confusion matrix display

        # Formatting for the plot
        if labels:
            xticks_rotation = 'vertical'
        else:
            xticks_rotation = 'horizontal'

        display.plot(include_values=True, cmap=plt.cm.get_cmap('Blues'), xticks_rotation=xticks_rotation, ax=ax)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.close()


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists_two_branch(train_class_num,trainloader,device,net,alpha=0.2,cache_dir=None):
    scores_0 = [[] for _ in range(train_class_num)]
    scores_1 = [[] for _ in range(train_class_num)]
    scores = [[] for _ in range(train_class_num)]

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train:'

    output0 = None
    output1 = None
    targets = None

    output0_path = osp.join(cache_dir, "train_output0_openset_embed.npy")
    output1_path = osp.join(cache_dir, "train_output1_openset_embed.npy")
    targets_path = osp.join(cache_dir, "train_targets_openset_labels.npy")
    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []
        with torch.no_grad():
            for images, target in metric_logger.log_every(trainloader, 10, header):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = net(images)
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
            output0 = torch.cat(output0, dim=0)
            output1 = torch.cat(output1, dim=0)
            targets = torch.cat(targets, dim=0)
            if utils.is_main_process():
                np.save(output0_path, output0.cpu().numpy())
                np.save(output1_path, output1.cpu().numpy())
                np.save(targets_path, targets.cpu().numpy())

    for score_0, score_1, t in zip(output0, output1, targets):
        if torch.argmax(score_0) == t:
            scores_0[t].append(score_0.unsqueeze(dim=0).unsqueeze(dim=0))
        if torch.argmax(score_1) == t:
            scores_1[t].append(score_1.unsqueeze(dim=0).unsqueeze(dim=0))
        score = score_0.softmax(0) * alpha + score_1.softmax(0) * (1 - alpha)
        if torch.argmax(score) == t:
            scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    scores_0 = [torch.cat(x).cpu().numpy() for x in scores_0]  # (N_c, 1, C) * C
    scores_1 = [torch.cat(x).cpu().numpy() for x in scores_1]  # (N_c, 1, C) * C
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs_0 = np.array([np.mean(x, axis=0) for x in scores_0])  # (C, 1, C)
    mavs_1 = np.array([np.mean(x, axis=0) for x in scores_1])  # (C, 1, C)
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists_0 = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs_0, scores_0)]
    dists_1 = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs_1, scores_1)]
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores_0, mavs_0, dists_0, scores_1, mavs_1, dists_1, scores, mavs, dists


def compute_train_score_and_mavs_and_dists_two_branch_dist(train_class_num,trainloader,device,net,alpha=0.2,cache_dir=None):
    scores_0 = [[] for _ in range(train_class_num)]
    scores_1 = [[] for _ in range(train_class_num)]
    scores = [[] for _ in range(train_class_num)]

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train:'

    output0 = None
    output1 = None
    targets = None

    output0_path = osp.join(cache_dir, "train_output0_openset_embed_dist.npy")
    output1_path = osp.join(cache_dir, "train_output1_openset_embed_dist.npy")
    targets_path = osp.join(cache_dir, "train_targets_openset_labels_dist.npy")

    total_size = len(trainloader.dataset)

    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    rank = utils.get_rank()

    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []
        with torch.no_grad():
            for inputs, target in metric_logger.log_every(trainloader, 10, header):
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = net(inputs)
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
            output0 = torch.cat(output0, dim=0)
            output1 = torch.cat(output1, dim=0)
            targets = torch.cat(targets, dim=0)

        src = output0[1]
        if dist.is_initialized():
            out = [torch.zeros_like(output0.contiguous()) for _ in range(dist.get_world_size())]
            dist.all_gather(out, output0.contiguous())
            output0 = tuple(out)
            world_size = utils.get_world_size()
            ordered_results = []
            for res in zip(*output0):
                ordered_results.extend(list(res))
            output0 = ordered_results[:total_size]
            output0 = torch.stack(output0, dim=0).to(device)
            assert torch.equal(src, output0[world_size + rank])
        if utils.is_main_process():
            np.save(output0_path, output0.cpu().numpy())

        src = output1[1]
        if dist.is_initialized():
            out = [torch.zeros_like(output1.contiguous()) for _ in range(dist.get_world_size())]
            dist.all_gather(out, output1.contiguous())
            output1 = tuple(out)
            world_size = utils.get_world_size()
            ordered_results = []
            for res in zip(*output1):
                ordered_results.extend(list(res))
            output1 = ordered_results[:total_size]
            output1 = torch.stack(output1, dim=0).to(device)
            assert torch.equal(src, output1[world_size + rank])
        if utils.is_main_process():
            np.save(output1_path, output1.cpu().numpy())

        src = targets[1]
        if dist.is_initialized():
            out = [torch.zeros_like(targets.contiguous()) for _ in range(dist.get_world_size())]
            dist.all_gather(out, targets.contiguous())
            targets = tuple(out)
            world_size = utils.get_world_size()
            ordered_results = []
            for res in zip(*targets):
                ordered_results.extend(list(res))
            targets = ordered_results[:total_size]
            targets = torch.stack(targets, dim=0).to(device)
            assert torch.equal(src, targets[world_size + rank])
        if utils.is_main_process():
            np.save(targets_path, targets.cpu().numpy())

    for score_0, score_1, t in zip(output0, output1, targets):
        if torch.argmax(score_0) == t:
            scores_0[t].append(score_0.unsqueeze(dim=0).unsqueeze(dim=0))
        if torch.argmax(score_1) == t:
            scores_1[t].append(score_1.unsqueeze(dim=0).unsqueeze(dim=0))
        score = score_0.softmax(0) * alpha + score_1.softmax(0) * (1 - alpha)
        if torch.argmax(score) == t:
            scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    for i, (s1, s2) in enumerate(zip(scores_0, scores_1)):
        if len(s1) == 0 and len(s2) != 0:
            scores_0[i] = s2
        elif len(s1) != 0 and len(s2) == 0:
            scores_1[i] = s1[:1]
        elif len(s1) == 0 and len(s2) == 0:
            print(f's1 and s2 are all o0 in {i}')

    scores_0 = [torch.cat(x).cpu().numpy() for x in scores_0]  # (N_c, 1, C) * C
    scores_1 = [torch.cat(x).cpu().numpy() for x in scores_1]  # (N_c, 1, C) * C
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs_0 = np.array([np.mean(x, axis=0) for x in scores_0])  # (C, 1, C)
    mavs_1 = np.array([np.mean(x, axis=0) for x in scores_1])  # (C, 1, C)
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists_0 = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs_0, scores_0)]
    dists_1 = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs_1, scores_1)]
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores_0, mavs_0, dists_0, scores_1, mavs_1, dists_1, scores, mavs, dists
