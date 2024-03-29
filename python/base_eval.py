import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
import open3d as o3d
from time import time
from random import seed
import argparse
import scipy.io as sio
from sklearn.neighbors import KDTree
from tqdm import tqdm

seed(5)
np.random.seed(5)
milliseconds = int(time() * 1000)

# argument parsing
parse = argparse.ArgumentParser(description="Depth Map Fusion Deep Network evaluation.")

parse.add_argument("-m", "--method", default="colmap", type=str, help="Method name (e.x. colmap).")
parse.add_argument("-l", "--light_setting", default="l3", type=str, help="DTU light setting.")
parse.add_argument("-p", "--representation", default="Points", type=str, help="Data representation (Points/Surface).")
parse.add_argument("-d", "--data_path", default="../../mvs_data", type=str, help="Path to the DTU evaluation data.")
parse.add_argument("-o", "--output_path", default="../../Output", type=str, help="Output results path where the output evaluation metrics will be stored.")
parse.add_argument("-s", "--points_path", default="../../Points", type=str)
parse.add_argument("-y", "--ply_file_name", default="output.ply", type=str, help="File name for the estimated point clouds.")
parse.add_argument("-e", "--eval_list", default="1,9,23,77,114", type=str, help="Scene evaluation list following the format '#,#,#,#' (e.x. '1,9,23,77,114') COMMA-SEPARATED, NO-SPACES.")
parse.add_argument("-q", "--min_point_dist", default=0.20, type=float, help="Minimum distance between points for downsampling.")
parse.add_argument("-x", "--max_dist", default=0.4, type=float, help="Max distance threshold for point matching.")
parse.add_argument("-k", "--mask_th", default=20.0, type=float, help="Max distance for masking.")
parse.add_argument("-n", "--min_dist", default=0.0, type=float, help="Min distance threshold for point matching")

ARGS = parse.parse_args()


def true_round(n):
    return np.round(n+0.5)

def save_ply(file_path, ply):
    o3d.io.write_point_cloud(file_path, ply)

def read_point_cloud(ply_path):
    if(ply_path[-3:] != "ply"):
        print("Error: file {} is not a '.ply' file.".format(ply_path))

    return o3d.io.read_point_cloud(ply_path, format="ply")

def downsample_cloud(cloud, min_point_dist):
    return cloud.voxel_down_sample(voxel_size=min_point_dist)

def remove_close_points(ply, min_point_dist):
    # build KD-Tree of estimated point cloud for querying
    tree = KDTree(np.asarray(ply.points), leaf_size=40)
    (dists, inds) = tree.query(np.asarray(ply.points), k=2)

    # ignore first nearest neighbor (since it is the point itself)
    dists = dists[:,1]
    inds = inds[:,1]

    # get unique set of indices
    ind_pairs = np.asarray([ np.asarray([min(i,inds[i]),max(i,inds[i])]) for i in range(len(inds)) ])
    close_inds = np.where(dists < min_point_dist)[0]
    print(close_inds.shape)
    remove_inds = set()
    pair_set = set()
    
    for ind in tqdm(close_inds):
        i,j = ind_pairs[ind]

        if ((i,j) not in pair_set):
            remove_inds.add(i)
            pair_set.add((i,j))

    remove_inds = list(remove_inds)

    #   for ind in tqdm(close_inds):
    #       # check if index is in index pair list
    #       same_inds = np.where(ind_pairs == ind)[0]
    #       if (len(same_inds) > 0):
    #           # add index to unique removal index list
    #           unique_inds.append(ind)

    #           for offset,i in enumerate(same_inds):
    #               index = i - offset
    #               ind_pairs = np.delete(ind_pairs, index, axis=0)


    cloud = ply.select_by_index(remove_inds, invert=True)
    print(len(cloud.points))

    return cloud

def build_est_points_filter(est_ply, data_path, scan_num):
    # read in matlab bounding box, mask, and resolution
    mask_filename = "ObsMask{}_10.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    bounds = np.asarray(data["BB"])
    min_bound = bounds[0,:]
    max_bound = bounds[1,:]
    mask = np.asarray(data["ObsMask"])
    res = int(data["Res"])

    points = np.asarray(est_ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = true_round(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_gt_points_filter(ply, data_path, scan_num):
    # read in matlab gt plane 
    mask_filename = "Plane{}.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    P = np.asarray(data["P"])

    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt

def filter_outlier_points(est_ply, gt_ply, outlier_th):
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))
    valid_dists = np.where(dists_est <= outlier_th)[0]
    return est_ply.select_by_index(valid_dists)

def compare_point_clouds(est_ply, gt_ply, mask_th, max_dist, min_dist, est_filt=None, gt_filt=None):
    mask_gt = 20.0
    inlier_th = 0.5

    # compute bi-directional chamfer distance between point clouds
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))
    valid_inds_est = set(np.where(est_filt == 1)[0])
    valid_dists = set(np.where(dists_est <= mask_th)[0])
    valid_inds_est.intersection_update(valid_dists)
    inlier_inds_est = set(np.where(dists_est < inlier_th)[0])
    inlier_inds_est.intersection_update(valid_inds_est)
    outlier_inds_est = set(np.where(dists_est >= inlier_th)[0])
    outlier_inds_est.intersection_update(valid_inds_est)
    valid_inds_est = np.asarray(list(valid_inds_est))
    inlier_inds_est = np.asarray(list(inlier_inds_est))
    outlier_inds_est = np.asarray(list(outlier_inds_est))
    dists_est = dists_est[valid_inds_est]

    dists_gt = np.asarray(gt_ply.compute_point_cloud_distance(est_ply))
    valid_inds_gt = set(np.where(gt_filt == 1)[0])
    valid_dists = set(np.where(dists_gt <= mask_gt)[0])
    valid_inds_gt.intersection_update(valid_dists)
    inlier_inds_gt = set(np.where(dists_gt < inlier_th)[0])
    inlier_inds_gt.intersection_update(valid_inds_gt)
    outlier_inds_gt = set(np.where(dists_gt >= inlier_th)[0])
    outlier_inds_gt.intersection_update(valid_inds_gt)
    valid_inds_gt = np.asarray(list(valid_inds_gt))
    inlier_inds_gt = np.asarray(list(inlier_inds_gt))
    outlier_inds_gt = np.asarray(list(outlier_inds_gt))
    dists_gt = dists_gt[valid_inds_gt]

    # compute accuracy and competeness
    acc = np.mean(dists_est)
    comp = np.mean(dists_gt)
    #   perc_th = 2.0
    #   acc = (len(np.where(dists_est <= perc_th)[0]) / len(dists_est))
    #   comp = (len(np.where(dists_gt <= perc_th)[0]) / len(dists_gt))



    # measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3*max_dist, num=50)
    prec_vals = [ (len(np.where(dists_est <= th)[0]) / len(dists_est)) for th in th_vals ]
    rec_vals = [ (len(np.where(dists_gt <= th)[0]) / len(dists_gt)) for th in th_vals ]

    # compute precision and recall for given distance threshold
    prec = len(np.where(dists_est <= max_dist)[0]) / len(dists_est)
    rec = len(np.where(dists_gt <= max_dist)[0]) / len(dists_gt)

    # color point cloud for precision
    valid_est_ply = est_ply.select_by_index(valid_inds_est)
    est_size = len(valid_est_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_est, max_dist) / max_dist)[:, :3]
    valid_est_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points precision
    invalid_est_ply = est_ply.select_by_index(valid_inds_est, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_est_ply.points)))[:, :3]
    invalid_est_ply.colors = o3d.utility.Vector3dVector(colors)

    # color point cloud for recall
    valid_gt_ply = gt_ply.select_by_index(valid_inds_gt)
    gt_size = len(valid_gt_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_gt, max_dist) / max_dist)[:, :3]
    valid_gt_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points recall
    invalid_gt_ply = gt_ply.select_by_index(valid_inds_gt, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_gt_ply.points)))[:, :3]
    invalid_gt_ply.colors = o3d.utility.Vector3dVector(colors)

    # color accuracy outliers
    inlier_est_ply = est_ply.select_by_index(inlier_inds_est)
    outlier_est_ply = est_ply.select_by_index(outlier_inds_est)
    inlier_cmap = plt.get_cmap("binary")
    outlier_cmap = plt.get_cmap("cool")
    inlier_colors = inlier_cmap(np.ones(len(inlier_est_ply.points)))[:, :3]
    outlier_colors = outlier_cmap(np.ones(len(outlier_est_ply.points)))[:, :3]
    inlier_est_ply.colors = o3d.utility.Vector3dVector(inlier_colors)
    outlier_est_ply.colors = o3d.utility.Vector3dVector(outlier_colors)

    # color completeness outliers
    inlier_gt_ply = gt_ply.select_by_index(inlier_inds_gt)
    outlier_gt_ply = gt_ply.select_by_index(outlier_inds_gt)
    inlier_cmap = plt.get_cmap("binary")
    outlier_cmap = plt.get_cmap("cool")
    inlier_colors = inlier_cmap(np.ones(len(inlier_gt_ply.points)))[:, :3]
    outlier_colors = outlier_cmap(np.ones(len(outlier_gt_ply.points)))[:, :3]
    inlier_gt_ply.colors = o3d.utility.Vector3dVector(inlier_colors)
    outlier_gt_ply.colors = o3d.utility.Vector3dVector(outlier_colors)

    precision_ply = valid_est_ply + invalid_est_ply
    recall_ply = valid_gt_ply + invalid_gt_ply
    acc_outliers_ply = inlier_est_ply+outlier_est_ply+invalid_est_ply
    comp_outliers_ply = inlier_gt_ply+outlier_gt_ply+invalid_gt_ply

    return  (precision_ply, recall_ply) \
            ,(acc_outliers_ply, comp_outliers_ply) \
            ,(acc,comp), (prec, rec) \
            ,(th_vals, prec_vals, rec_vals) \
            ,(est_size, gt_size)

def main():
    # convert eval list to integers
    eval_list = ARGS.eval_list.split(',')
    eval_list = [int(e) for e in eval_list]
    num_evals = len(eval_list)

    # variables for recording averages
    avg_est_size = 0
    avg_gt_size = 0
    avg_acc = 0.0
    avg_comp = 0.0
    avg_prec = 0.0
    avg_rec = 0.0
    avg_dur = 0.0

    for scan_num in eval_list:
        start_total = time()
        print("\nEvaluating scan{:03d}...".format(scan_num))

        # read in point clouds
        est_ply_filename = "{}{:03d}_l3.ply".format(ARGS.method,scan_num)
        points_path = os.path.join(ARGS.output_path, "scan{}".format(str(scan_num).zfill(3)), ARGS.points_path)
        est_ply_path = os.path.join(points_path, est_ply_filename)
        est_ply = read_point_cloud(est_ply_path)
        est_ply = downsample_cloud(est_ply, ARGS.min_point_dist)
        #est_ply = remove_close_points(est_ply, ARGS.min_point_dist)

        gt_ply_filename = "stl{:03d}_total.ply".format(scan_num)
        gt_ply_path = os.path.join(ARGS.data_path, "Points", "stl", gt_ply_filename)
        gt_ply = read_point_cloud(gt_ply_path)

        # build points filter based on input mask
        est_ply = filter_outlier_points(est_ply, gt_ply, ARGS.mask_th)
        est_filt = build_est_points_filter(est_ply, ARGS.data_path, scan_num)
        gt_filt = build_gt_points_filter(gt_ply, ARGS.data_path, scan_num)

        # compute distances between point clouds
        (precision_ply, recall_ply) \
        ,(acc_outliers_ply, comp_outliers_ply) \
        ,(acc,comp) \
        ,(prec, rec) \
        ,(th_vals, prec_vals, rec_vals) \
        ,(est_size, gt_size) = \
        compare_point_clouds(est_ply, gt_ply, ARGS.mask_th, ARGS.max_dist, ARGS.min_dist, est_filt, gt_filt)

        end_total = time()
        dur = end_total-start_total

        # display current evaluation
        print("Num Est: {}".format(int(est_size)))
        print("Num GT: {}".format(int(gt_size)))
        print("Accuracy: {:0.4f}".format(acc))
        print("Completeness: {:0.4f}".format(comp))
        print("Overall: {:0.4f}".format((acc+comp)/2.0))
        print("Precision: {:0.4f}".format(prec))
        print("Recall: {:0.4f}".format(rec))
        print("Elapsed time: {:0.3f} s".format(dur))


        ##### Save metrics #####
        eval_path = os.path.join(points_path, "eval")
        if (os.path.exists(eval_path)):
            shutil.rmtree(eval_path)
        os.mkdir(eval_path)

        # save precision point cloud
        precision_path = os.path.join(eval_path, "precision.ply")
        save_ply(precision_path, precision_ply)

        # save recall point cloud
        recall_path = os.path.join(eval_path, "recall.ply")
        save_ply(recall_path, recall_ply)

        # save binary accuracy point cloud
        acc_outliers_path = os.path.join(eval_path, "acc_outliers.ply")
        save_ply(acc_outliers_path, acc_outliers_ply)

        # save recall point cloud
        comp_outliers_path = os.path.join(eval_path, "comp_outliers.ply")
        save_ply(comp_outliers_path, comp_outliers_ply)

        # create plots for incremental threshold values
        plot_filename = os.path.join(eval_path, "eval.png")
        plt.plot(th_vals, prec_vals, th_vals, rec_vals)
        plt.title("Precision and Recall (t={}mm)".format(ARGS.max_dist))
        plt.xlabel("threshold")
        plt.vlines(ARGS.max_dist, 0, 1, linestyles='dashed', label='t')
        plt.legend(("precision", "recall"))
        plt.grid()
        plt.savefig(plot_filename)
        plt.close()

        # write all metrics to the evaluation file
        stats_file = os.path.join(eval_path, "metrics.txt")
        with open(stats_file, 'w') as f:
            f.write("Method: {}\n".format(ARGS.method))
            f.write("Min_point_dist: {:0.3f}mm | Max distance threshold: {:0.3f}mm | Min distance threshold: {:0.3f}mm | Mask threshold: {:0.3f}mm\n".format(ARGS.min_point_dist, ARGS.max_dist, ARGS.min_dist, ARGS.mask_th))
            f.write("Source point cloud size: {}\n".format(est_size))
            f.write("Target point cloud size: {}\n".format(gt_size))
            f.write("Accuracy: {:0.3f}mm\n".format(acc))
            f.write("Completness: {:0.3f}mm\n".format(comp))
            f.write("Overall: {:0.4f}\n".format((acc+comp)/2.0))
            f.write("Precision: {:0.3f}\n".format(prec))
            f.write("Recall: {:0.3f}\n".format(rec))

        # record averages
        avg_est_size    += est_size
        avg_gt_size     += gt_size
        avg_acc         += acc
        avg_comp        += comp
        avg_prec        += prec
        avg_rec         += rec
        avg_dur         += dur

        # cleanup data
        del est_ply
        del gt_ply
        del precision_ply
        del recall_ply
        del acc_outliers_ply
        del comp_outliers_ply

    # display average evaluation
    print("\nAveraged evaluation..")
    print("Num Est: {}".format(int(avg_est_size // num_evals)))
    print("Num GT: {}".format(int(avg_gt_size // num_evals)))
    print("Accuracy: {:0.4f}".format(avg_acc / num_evals))
    print("Completeness: {:0.4f}".format(avg_comp / num_evals))
    print("Overall: {:0.4f}".format(((avg_acc/num_evals)+(avg_comp/num_evals))/2.0))
    print("Precision: {:0.4f}".format(avg_prec / num_evals))
    print("Recall: {:0.4f}".format(avg_rec / num_evals))
    print("Elapsed time: {:0.3f} s".format(avg_dur / num_evals))


if __name__=="__main__":
    main()
