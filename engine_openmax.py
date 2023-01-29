import os.path as osp

import torch
import torch.distributed as dist

import utils
import numpy as np
import os

from openmax import compute_train_score_and_mavs_and_dists_two_branch, fit_weibull, \
    openmax, Evaluation, compute_train_score_and_mavs_and_dists_two_branch_dist


def calc_openset_acc(pred_softmax, pred_softmax_threshold, pred_openmax, score_softmax, score_openmax, labels):
    eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
    eval_openmax = Evaluation(pred_openmax, labels, score_openmax)

    print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
    print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
    print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
    print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
    # print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))
    print(f"_________________________________________")

    print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
    print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
    print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
    # print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
    print(f"_________________________________________")

    print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
    print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
    print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
    # print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
    print(f"_________________________________________")

    return eval_softmax, eval_softmax_threshold, eval_openmax


@torch.no_grad()
def evaluate_openset(train_loader, test_loader, model, device, args=None, tokens=None):
    # we adopt the max f1_macro score from output0, output1, output
    two_branch = args.two_branch

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    output0 = None
    output1 = None
    targets = None

    assert two_branch

    cache_dir = osp.dirname(args.resume)
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        output0_path = osp.join(cache_dir, "test_output0_openset_embed.npy")
        output1_path = osp.join(cache_dir, "test_output1_openset_embed.npy")
        targets_path = osp.join(cache_dir, "test_targets_openset_labels.npy")
    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []

        for images, target in metric_logger.log_every(test_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            inputs = (images, texts) if texts is not None else images
            # compute output
            with torch.cuda.amp.autocast():
                output = model(inputs)

                if two_branch:
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
                else:
                    output0.append(output)
                    targets.append(targets)
        output0 = torch.cat(output0, dim=0)
        output1 = torch.cat(output1, dim=0)
        targets = torch.cat(targets, dim=0)
        if utils.is_main_process():
            np.save(output0_path, output0.cpu().numpy())
            np.save(output1_path, output1.cpu().numpy())
            np.save(targets_path, targets.cpu().numpy())

    if two_branch:
        alpha = args.alpha
        print(f'alpha: {alpha}')
        output = output0.softmax(1) * alpha + output1.softmax(1) * (1 - alpha)
        output = output.cpu().numpy()
        output0 = output0.cpu().numpy()
        output1 = output1.cpu().numpy()
        targets = targets.cpu().numpy()

        output0 = np.array(output0)[:, np.newaxis, :]
        output1 = np.array(output1)[:, np.newaxis, :]
        output = np.array(output)[:, np.newaxis, :]
        labels = np.array(targets)

        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")
        _, mavs_0, dists_0, _, mavs_1, dists_1, _, mavs, dists = \
            compute_train_score_and_mavs_and_dists_two_branch(args.train_class_num, train_loader, device, model, alpha,
                                                              cache_dir)
        categories = list(range(0, args.train_class_num))
        weibull_model_0 = fit_weibull(mavs_0, dists_0, categories, args.weibull_tail, "euclidean")
        weibull_model_1 = fit_weibull(mavs_1, dists_1, categories, args.weibull_tail, "euclidean")
        weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

        print("Evaluation output0 ...")
        pred_softmax_0, pred_softmax_threshold_0, pred_openmax_0 = [], [], []
        score_softmax_0, score_openmax_0 = [], []
        for score in output0:
            so, ss = openmax(weibull_model_0, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_0.append(np.argmax(ss))
            pred_softmax_threshold_0.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_0.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_0.append(ss)
            score_openmax_0.append(so)

        eval_softmax_0, eval_softmax_threshold_0, eval_openmax_0 = calc_openset_acc(pred_softmax_0,
                                                                                    pred_softmax_threshold_0,
                                                                                    pred_openmax_0,
                                                                                    score_softmax_0,
                                                                                    score_openmax_0,
                                                                                    labels)

        torch.save(eval_softmax_0, os.path.join(cache_dir, 'output0_v1_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold_0, os.path.join(cache_dir, 'output0_v1_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax_0, os.path.join(cache_dir, 'output0_v1_eval_openmax.pkl'))

        print("Evaluation output1 ...")
        pred_softmax_1, pred_softmax_threshold_1, pred_openmax_1 = [], [], []
        score_softmax_1, score_openmax_1 = [], []
        for score in output1:
            so, ss = openmax(weibull_model_1, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_1.append(np.argmax(ss))
            pred_softmax_threshold_1.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_1.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_1.append(ss)
            score_openmax_1.append(so)

        eval_softmax_1, eval_softmax_threshold_1, eval_openmax_1 = calc_openset_acc(pred_softmax_1,
                                                                                    pred_softmax_threshold_1,
                                                                                    pred_openmax_1,
                                                                                    score_softmax_1,
                                                                                    score_openmax_1,
                                                                                    labels)

        torch.save(eval_softmax_1, os.path.join(cache_dir, 'output1_v1_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold_1, os.path.join(cache_dir, 'output1_v1_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax_1, os.path.join(cache_dir, 'output1_v1_eval_openmax.pkl'))

        print("Evaluation output ...")
        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []
        for score in output:
            so, ss = openmax(weibull_model, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax.append(ss)
            score_openmax.append(so)

        eval_softmax, eval_softmax_threshold, eval_openmax = calc_openset_acc(pred_softmax,
                                                                              pred_softmax_threshold,
                                                                              pred_openmax,
                                                                              score_softmax,
                                                                              score_openmax,
                                                                              labels)

        torch.save(eval_softmax, os.path.join(cache_dir, 'output_v1_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold, os.path.join(cache_dir, 'output_v1_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax, os.path.join(cache_dir, 'output_v1_eval_openmax.pkl'))


@torch.no_grad()
def evaluate_openset_v2(train_loader, test_loader, model, device, args=None, tokens=None):
    # we adopt the max f1_macro score from output0, output1, output
    two_branch = args.two_branch

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    output0 = None
    output1 = None
    targets = None

    assert two_branch

    cache_dir = osp.dirname(args.resume)
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        output0_path = osp.join(cache_dir, "test_output0_openset_v2_embed.npy")
        output1_path = osp.join(cache_dir, "test_output1_openset_v2_embed.npy")
        targets_path = osp.join(cache_dir, "test_targets_openset_v2_labels.npy")
    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []

        for images, target in metric_logger.log_every(test_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            inputs = (images, texts) if texts is not None else images
            # compute output
            with torch.cuda.amp.autocast():
                output = model(inputs)

                if two_branch:
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
                else:
                    output0.append(output)
                    targets.append(targets)
        output0 = torch.cat(output0, dim=0)
        output1 = torch.cat(output1, dim=0)
        targets = torch.cat(targets, dim=0)
        if utils.is_main_process():
            np.save(output0_path, output0.cpu().numpy())
            np.save(output1_path, output1.cpu().numpy())
            np.save(targets_path, targets.cpu().numpy())

    if two_branch:
        alpha = args.alpha
        print(f'alpha: {alpha}')
        output0 = output0.cpu().numpy()
        output1 = output1.cpu().numpy()
        targets = targets.cpu().numpy()

        output0 = np.array(output0)[:, np.newaxis, :]
        output1 = np.array(output1)[:, np.newaxis, :]
        # output = np.array(output)[:, np.newaxis, :]
        labels = np.array(targets)

        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")
        _, mavs_0, dists_0, _, mavs_1, dists_1, _, _, _ = \
            compute_train_score_and_mavs_and_dists_two_branch(args.train_class_num, train_loader, device, model, alpha,
                                                              cache_dir)
        categories = list(range(0, args.train_class_num))
        weibull_model_0 = fit_weibull(mavs_0, dists_0, categories, args.weibull_tail, "euclidean")
        weibull_model_1 = fit_weibull(mavs_1, dists_1, categories, args.weibull_tail, "euclidean")
        # weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

        print("Evaluation output0 ...")
        pred_softmax_0, pred_softmax_threshold_0, pred_openmax_0 = [], [], []
        score_softmax_0, score_openmax_0 = [], []
        for score in output0:
            so, ss = openmax(weibull_model_0, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_0.append(np.argmax(ss))
            pred_softmax_threshold_0.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_0.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_0.append(ss)
            score_openmax_0.append(so)

        eval_softmax_0, eval_softmax_threshold_0, eval_openmax_0 = calc_openset_acc(pred_softmax_0,
                                                                                    pred_softmax_threshold_0,
                                                                                    pred_openmax_0,
                                                                                    score_softmax_0,
                                                                                    score_openmax_0,
                                                                                    labels)

        torch.save(eval_softmax_0, os.path.join(cache_dir, 'output0_v2_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold_0, os.path.join(cache_dir, 'output0_v2_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax_0, os.path.join(cache_dir, 'output0_v2_eval_openmax_dist.pkl'))

        print("Evaluation output1 ...")
        pred_softmax_1, pred_softmax_threshold_1, pred_openmax_1 = [], [], []
        score_softmax_1, score_openmax_1 = [], []
        for score in output1:
            so, ss = openmax(weibull_model_1, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_1.append(np.argmax(ss))
            pred_softmax_threshold_1.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_1.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_1.append(ss)
            score_openmax_1.append(so)

        eval_softmax_1, eval_softmax_threshold_1, eval_openmax_1 = calc_openset_acc(pred_softmax_1,
                                                                                    pred_softmax_threshold_1,
                                                                                    pred_openmax_1,
                                                                                    score_softmax_1,
                                                                                    score_openmax_1,
                                                                                    labels)

        torch.save(eval_softmax_1, os.path.join(cache_dir, 'output1_v2_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold_1, os.path.join(cache_dir, 'output1_v2_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax_1, os.path.join(cache_dir, 'output1_v2_eval_openmax_dist.pkl'))

        print("Evaluation output ...")
        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []

        for i in range(len(score_openmax_1)):
            so_0, ss_0, so_1, ss_1 = score_openmax_0[i], score_softmax_0[i], score_openmax_1[i], score_softmax_1[i]
            so = so_0 * alpha + so_1 * (1 - alpha)
            ss = ss_0 * alpha + ss_1 * (1 - alpha)
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax.append(ss)
            score_openmax.append(so)

        eval_softmax, eval_softmax_threshold, eval_openmax = calc_openset_acc(pred_softmax,
                                                                              pred_softmax_threshold,
                                                                              pred_openmax,
                                                                              score_softmax,
                                                                              score_openmax,
                                                                              labels)

        torch.save(eval_softmax, os.path.join(cache_dir, 'output_v2_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold, os.path.join(cache_dir, 'output_v2_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax, os.path.join(cache_dir, 'output_v2_eval_openmax_dist.pkl'))


@torch.no_grad()
def evaluate_openset_dist(train_loader, test_loader, model, device, args=None, tokens=None):
    # we adopt the max f1_macro score from output0, output1, output
    two_branch = args.two_branch

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    output0 = None
    output1 = None
    targets = None

    assert two_branch

    cache_dir = osp.dirname(args.resume)
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        output0_path = osp.join(cache_dir, "test_output0_openset_embed_dist.npy")
        output1_path = osp.join(cache_dir, "test_output1_openset_embed_dist.npy")
        targets_path = osp.join(cache_dir, "test_targets_openset_labels_dist.npy")

    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    rank = utils.get_rank()
    test_total_size = len(test_loader.dataset)
    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []

        for images, target in metric_logger.log_every(test_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            inputs = (images, texts) if texts is not None else images

            with torch.cuda.amp.autocast():
                output = model(inputs)

                if two_branch:
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
                else:
                    output0.append(output)
                    targets.append(targets)
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
            output0 = ordered_results[:test_total_size]
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
            output1 = ordered_results[:test_total_size]
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
            targets = ordered_results[:test_total_size]
            targets = torch.stack(targets, dim=0).to(device)
            assert torch.equal(src, targets[world_size + rank])
        if utils.is_main_process():
            np.save(targets_path, targets.cpu().numpy())

    dist.barrier()

    if two_branch:
        alpha = args.alpha
        output = output0.softmax(1) * alpha + output1.softmax(1) * (1 - alpha)
        output = output.cpu().numpy()
        output0 = output0.cpu().numpy()
        output1 = output1.cpu().numpy()
        targets = targets.cpu().numpy()

        output0 = np.array(output0)[:, np.newaxis, :]
        output1 = np.array(output1)[:, np.newaxis, :]
        output = np.array(output)[:, np.newaxis, :]
        labels = np.array(targets)

        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")
        _, mavs_0, dists_0, _, mavs_1, dists_1, _, mavs, dists = \
            compute_train_score_and_mavs_and_dists_two_branch_dist(args.train_class_num, train_loader, device, model,
                                                                   alpha,
                                                                   cache_dir)
        categories = list(range(0, args.train_class_num))
        if rank != 0:
            return
        weibull_model_0 = fit_weibull(mavs_0, dists_0, categories, args.weibull_tail, "euclidean")
        weibull_model_1 = fit_weibull(mavs_1, dists_1, categories, args.weibull_tail, "euclidean")
        weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

        print("Evaluation output0 ...")
        pred_softmax_0, pred_softmax_threshold_0, pred_openmax_0 = [], [], []
        score_softmax_0, score_openmax_0 = [], []
        for score in output0:
            so, ss = openmax(weibull_model_0, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_0.append(np.argmax(ss))
            pred_softmax_threshold_0.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_0.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_0.append(ss)
            score_openmax_0.append(so)

        eval_softmax_0, eval_softmax_threshold_0, eval_openmax_0 = calc_openset_acc(pred_softmax_0,
                                                                                    pred_softmax_threshold_0,
                                                                                    pred_openmax_0,
                                                                                    score_softmax_0,
                                                                                    score_openmax_0,
                                                                                    labels)

        torch.save(eval_softmax_0, os.path.join(cache_dir, 'output0_v1_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold_0, os.path.join(cache_dir, 'output0_v1_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax_0, os.path.join(cache_dir, 'output0_v1_eval_openmax_dist.pkl'))

        print("Evaluation output1 ...")
        pred_softmax_1, pred_softmax_threshold_1, pred_openmax_1 = [], [], []
        score_softmax_1, score_openmax_1 = [], []
        for score in output1:
            so, ss = openmax(weibull_model_1, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_1.append(np.argmax(ss))
            pred_softmax_threshold_1.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_1.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_1.append(ss)
            score_openmax_1.append(so)

        eval_softmax_1, eval_softmax_threshold_1, eval_openmax_1 = calc_openset_acc(pred_softmax_1,
                                                                                    pred_softmax_threshold_1,
                                                                                    pred_openmax_1,
                                                                                    score_softmax_1,
                                                                                    score_openmax_1,
                                                                                    labels)

        torch.save(eval_softmax_1, os.path.join(cache_dir, 'output1_v1_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold_1, os.path.join(cache_dir, 'output1_v1_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax_1, os.path.join(cache_dir, 'output1_v1_eval_openmax_dist.pkl'))

        print("Evaluation output ...")
        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []
        for score in output:
            so, ss = openmax(weibull_model, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax.append(ss)
            score_openmax.append(so)

        eval_softmax, eval_softmax_threshold, eval_openmax = calc_openset_acc(pred_softmax,
                                                                              pred_softmax_threshold,
                                                                              pred_openmax,
                                                                              score_softmax,
                                                                              score_openmax,
                                                                              labels)

        torch.save(eval_softmax, os.path.join(cache_dir, 'output_v1_eval_softmax_dist.pkl'))
        torch.save(eval_softmax_threshold, os.path.join(cache_dir, 'output_v1_eval_softmax_threshold_dist.pkl'))
        torch.save(eval_openmax, os.path.join(cache_dir, 'output_v1_eval_openmax_dist.pkl'))


@torch.no_grad()
def evaluate_openset_v2_dist(train_loader, test_loader, model, device, args=None, tokens=None):
    # we adopt the max f1_macro score from output0, output1, output
    two_branch = args.two_branch

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    output0 = None
    output1 = None
    targets = None

    assert two_branch

    cache_dir = osp.dirname(args.resume)
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        output0_path = osp.join(cache_dir, "test_output0_openset_v2_embed_dist.npy")
        output1_path = osp.join(cache_dir, "test_output1_openset_v2_embed_dist.npy")
        targets_path = osp.join(cache_dir, "test_targets_openset_v2_labels_dist.npy")
    if osp.exists(output0_path) and osp.exists(output1_path) and osp.exists(targets_path):
        print("using cached embeddings")
        output0 = torch.from_numpy(np.load(output0_path)).to(device, non_blocking=True)
        output1 = torch.from_numpy(np.load(output1_path)).to(device, non_blocking=True)
        targets = torch.from_numpy(np.load(targets_path)).to(device, non_blocking=True)

    rank = utils.get_rank()
    test_total_size = len(test_loader.dataset)
    if output0 is None and output1 is None and targets is None:
        output0 = []
        output1 = []
        targets = []

        for images, target in metric_logger.log_every(test_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            inputs = (images, texts) if texts is not None else images

            with torch.cuda.amp.autocast():
                output = model(inputs)

                if two_branch:
                    output0.append(output[0])
                    output1.append(output[1])
                    targets.append(target)
                else:
                    output0.append(output)
                    targets.append(targets)
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
            output0 = ordered_results[:test_total_size]
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
            output1 = ordered_results[:test_total_size]
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
            targets = ordered_results[:test_total_size]
            targets = torch.stack(targets, dim=0).to(device)
            assert torch.equal(src, targets[world_size + rank])
        if utils.is_main_process():
            np.save(targets_path, targets.cpu().numpy())

    dist.barrier()

    if two_branch:
        alpha = args.alpha
        output0 = output0.cpu().numpy()
        output1 = output1.cpu().numpy()
        targets = targets.cpu().numpy()

        output0 = np.array(output0)[:, np.newaxis, :]
        output1 = np.array(output1)[:, np.newaxis, :]
        labels = np.array(targets)

        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")
        _, mavs_0, dists_0, _, mavs_1, dists_1, _, _, _ = \
            compute_train_score_and_mavs_and_dists_two_branch_dist(args.train_class_num, train_loader, device, model,
                                                                   alpha,
                                                                   cache_dir)
        categories = list(range(0, args.train_class_num))
        if rank != 0:
            return

        weibull_model_0 = fit_weibull(mavs_0, dists_0, categories, args.weibull_tail, "euclidean")
        weibull_model_1 = fit_weibull(mavs_1, dists_1, categories, args.weibull_tail, "euclidean")
        # weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

        print("Evaluation output0 ...")
        pred_softmax_0, pred_softmax_threshold_0, pred_openmax_0 = [], [], []
        score_softmax_0, score_openmax_0 = [], []
        for score in output0:
            so, ss = openmax(weibull_model_0, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_0.append(np.argmax(ss))
            pred_softmax_threshold_0.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_0.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_0.append(ss)
            score_openmax_0.append(so)

        eval_softmax_0, eval_softmax_threshold_0, eval_openmax_0 = calc_openset_acc(pred_softmax_0,
                                                                                    pred_softmax_threshold_0,
                                                                                    pred_openmax_0,
                                                                                    score_softmax_0,
                                                                                    score_openmax_0,
                                                                                    labels)

        torch.save(eval_softmax_0, os.path.join(cache_dir, 'output0_v2_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold_0, os.path.join(cache_dir, 'output0_v2_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax_0, os.path.join(cache_dir, 'output0_v2_eval_openmax.pkl'))

        print("Evaluation output1 ...")
        pred_softmax_1, pred_softmax_threshold_1, pred_openmax_1 = [], [], []
        score_softmax_1, score_openmax_1 = [], []
        for score in output1:
            so, ss = openmax(weibull_model_1, categories, score,
                             0.5, args.weibull_alpha, "euclidean")
            # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
            pred_softmax_1.append(np.argmax(ss))
            pred_softmax_threshold_1.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax_1.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax_1.append(ss)
            score_openmax_1.append(so)

        eval_softmax_1, eval_softmax_threshold_1, eval_openmax_1 = calc_openset_acc(pred_softmax_1,
                                                                                    pred_softmax_threshold_1,
                                                                                    pred_openmax_1,
                                                                                    score_softmax_1,
                                                                                    score_openmax_1,
                                                                                    labels)

        torch.save(eval_softmax_1, os.path.join(cache_dir, 'output1_v2_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold_1, os.path.join(cache_dir, 'output1_v2_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax_1, os.path.join(cache_dir, 'output1_v2_eval_openmax.pkl'))

        print("Evaluation output ...")
        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []

        for i in range(len(score_openmax_1)):
            so_0, ss_0, so_1, ss_1 = score_openmax_0[i], score_softmax_0[i], score_openmax_1[i], score_softmax_1[i]
            so = so_0 * alpha + so_1 * (1 - alpha)
            ss = ss_0 * alpha + ss_1 * (1 - alpha)
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(
                np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
            pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
            score_softmax.append(ss)
            score_openmax.append(so)

        eval_softmax, eval_softmax_threshold, eval_openmax = calc_openset_acc(pred_softmax,
                                                                              pred_softmax_threshold,
                                                                              pred_openmax,
                                                                              score_softmax,
                                                                              score_openmax,
                                                                              labels)

        torch.save(eval_softmax, os.path.join(cache_dir, 'output_v2_eval_softmax.pkl'))
        torch.save(eval_softmax_threshold, os.path.join(cache_dir, 'output_v2_eval_softmax_threshold.pkl'))
        torch.save(eval_openmax, os.path.join(cache_dir, 'output_v2_eval_openmax.pkl'))
