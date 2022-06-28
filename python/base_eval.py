import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import open3d as o3d
from time import time
import scipy.io as sio

milliseconds = int(time() * 1000)

def read_point_cloud(ply_path):
    if(ply_path[-3:] != "ply"):
        print("Error: file {} is not a '.ply' file.".format(ply_path))

    ply = o3d.io.read_point_cloud(ply_path)

    return ply

def downsample_points(ply, min_dist):
    ply = ply.voxel_down_sample(voxel_size=min_dist)
    return ply

def build_est_points_filter(ply, min_bound, res, mask):
    points = np.asarray(ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res+1
    qv = np.round(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_gt_points_filter(ply, P):
    points = np.asarray(ply.points).transpose()
    shape = points.shape
    filt = np.zeros(shape[1])

    # compute iner-product between points and the defined plane
    Pt = P.transpose()
    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    in_bounds = np.asarray(np.where((plane_prod > 0))).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds] = 1

    return filt

def point_cloud_dist(src_ply, tgt_ply, max_dist, filt):
    dists = src_ply.compute_point_cloud_distance(tgt_ply)
    dists = np.asarray(dists)
    dists[dists > max_dist] = 0

    dists = dists * filt
    num_points = np.sum(filt)

    return num_points, np.mean(dists), np.var(dists), np.median(dists)

def compare_points(est_ply, gt_ply, data_path, max_dist, est_filt, gt_filt):
    # load mask, bounding box, and resolution
    (num_est, mean_acc, var_acc, med_acc) = point_cloud_dist(est_ply, gt_ply, max_dist, est_filt)
    (num_gt, mean_comp, var_comp, med_comp) = point_cloud_dist(gt_ply, est_ply, max_dist, gt_filt)

    return (num_est, num_gt, mean_acc, mean_comp, var_acc, var_comp, med_acc, med_comp)

def main():

    method = sys.argv[1]
    data_path = "../data"
    results_path = "../results"
    light_setting = "l3"
    representation = "Points"
    min_dist = 0.2
    max_dist = 60
    #eval_list = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    eval_list = [23]
    num_evals = len(eval_list)

    if (representation == "Points"):
        eval_string = "_Eval_IJCV_"
        settings_string = ""
    elif (representation == "Surfaces"):
        eval_string = "_SurfEval_Trim_IJCV_"
        settings_string = "_surf_11_trim_8"

    # variables for recording averages
    avg_num_est = 0
    avg_num_gt = 0
    avg_mean_acc = 0.0
    avg_mean_comp = 0.0
    avg_var_acc = 0.0
    avg_var_comp = 0.0
    avg_med_acc = 0.0
    avg_med_comp = 0.0

    for scan_num in eval_list:
        start = time()
        print("\nEvaluating scan{:03d}...".format(scan_num))

        # read in matlab bounding box, mask, and resolution
        mask_filename = "ObsMask{}_10.mat".format(scan_num)
        mask_path = os.path.join(data_path, "ObsMask", mask_filename)
        data = sio.loadmat(mask_path)
        bounds = np.asarray(data["BB"])
        min_bound = bounds[0,:]
        max_bound = bounds[1,:]
        mask = np.asarray(data["ObsMask"])
        res = int(data["Res"])

        # read in matlab gt plane 
        mask_filename = "Plane{}.mat".format(scan_num)
        mask_path = os.path.join(data_path, "ObsMask", mask_filename)
        data = sio.loadmat(mask_path)
        P = np.asarray(data["P"])

        # read in estimated point cloud
        est_ply_filename = "{}{:03d}_{}{}.ply".format(method, scan_num, light_setting, settings_string)
        est_ply_path = os.path.join(data_path, representation, method, est_ply_filename)
        est_ply = read_point_cloud(est_ply_path)
        est_ply = downsample_points(est_ply, min_dist)

        # read in ground-truth point cloud
        gt_ply_filename = "stl{:03d}_total.ply".format(scan_num)
        gt_ply_path = os.path.join(data_path, "Points", "ground_truth", gt_ply_filename)
        gt_ply = read_point_cloud(gt_ply_path) # already reduced to 0.2mm density, so no downsampling needed

        # crop estimated point cloud by gt bounding information
        #new_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        #est_ply = est_ply.crop(new_aabb)

        # build points filter based on input mask
        est_filt = build_est_points_filter(est_ply, min_bound, res, mask)

        # build points filter based on input mask
        gt_filt = build_gt_points_filter(gt_ply, P)

        # compute distances between point clouds
        (num_est, num_gt, mean_acc, mean_comp, var_acc, var_comp, med_acc, med_comp) = \
                compare_points(est_ply, gt_ply, data_path, max_dist, est_filt, gt_filt)

        end = time()
        print("Elapsed time: {:0.3f} s".format(end-start))

        # display current evaluation
        print("Num Est: {}".format(num_est))
        print("Num GT: {}".format(num_gt))
        print("Mean Acc: {:0.4f}".format(mean_acc))
        print("Mean Comp: {:0.4f}".format(mean_comp))
        print("Var Acc: {:0.4f}".format(var_acc))
        print("Var Comp: {:0.4f}".format(var_comp))
        print("Med Acc: {:0.4f}".format(med_acc))
        print("Med Comp: {:0.4f}".format(med_comp))

        # record averages
        avg_num_est     += num_est  
        avg_num_gt      += num_gt   
        avg_mean_acc    += mean_acc 
        avg_mean_comp   += mean_comp
        avg_var_acc     += var_acc  
        avg_var_comp    += var_comp 
        avg_med_acc     += med_acc  
        avg_med_comp    += med_comp 

    # display average evaluation
    print("\nAveraged evaluation..")
    print("Num Est: {}".format(avg_num_est // num_evals))
    print("Num GT: {}".format(avg_num_gt // num_evals))
    print("Mean Acc: {:0.4f}".format(avg_mean_acc / num_evals))
    print("Mean Comp: {:0.4f}".format(avg_mean_comp / num_evals))
    print("Var Acc: {:0.4f}".format(avg_var_acc / num_evals))
    print("Var Comp: {:0.4f}".format(avg_var_comp / num_evals))
    print("Med Acc: {:0.4f}".format(avg_med_acc / num_evals))
    print("Med Comp: {:0.4f}".format(avg_med_comp / num_evals))


if __name__=="__main__":
    main()
