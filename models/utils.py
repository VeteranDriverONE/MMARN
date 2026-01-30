# internal python imports
import os
import csv
import functools
import torch
import math

# third party imports
import numpy as np
import scipy
from skimage import measure
from pathlib import Path
import cv2

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端（无窗口弹出）
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互式模式（彻底禁用弹窗）

from PIL import Image
# local/our imports
import pystrum.pynd.ndutils as nd
from skimage.metrics import structural_similarity as ssim 
from scipy.ndimage import convolve  # 用于局部NCC计算
from models.losses import NCC

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.
    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    pairlist = [f.split(delim) for f in read_file_list(filename)]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist


def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.

    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi, ..., 0] = clean_seg(x[..., 0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[..., 0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize)
                for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def get_surface_pts_per_label(total_nb_surface_pts, layer_edge_ratios):
    """
    Gets the number of surface points per label, given the total number of surface points.
    """
    nb_surface_pts_sel = np.round(np.array(layer_edge_ratios) * total_nb_surface_pts).astype('int')
    nb_surface_pts_sel[-1] = total_nb_surface_pts - int(np.sum(nb_surface_pts_sel[:-1]))
    return nb_surface_pts_sel


def edge_to_surface_pts(X_edges, nb_surface_pts=None):
    """
    Converts edges to surface points.
    """

    # assumes X_edges is NOT in keras form
    surface_pts = np.stack(np.where(X_edges), 0).transpose()

    # random with replacements
    if nb_surface_pts is not None:
        chi = np.random.choice(range(surface_pts.shape[0]), size=nb_surface_pts)
        surface_pts = surface_pts[chi, :]

    return surface_pts


def sdt_to_surface_pts(X_sdt, nb_surface_pts,
                       surface_pts_upsample_factor=2, thr=0.50001, resize_fn=None):
    """
    Converts a signed distance transform to surface points.
    """
    us = [surface_pts_upsample_factor] * X_sdt.ndim

    if resize_fn is None:
        resized_vol = scipy.ndimage.interpolation.zoom(X_sdt, us, order=1, mode='reflect')
    else:
        resized_vol = resize_fn(X_sdt)
        pred_shape = np.array(X_sdt.shape) * surface_pts_upsample_factor
        assert np.array_equal(pred_shape, resized_vol.shape), 'resizing failed'

    X_edges = np.abs(resized_vol) < thr
    sf_pts = edge_to_surface_pts(X_edges, nb_surface_pts=nb_surface_pts)

    # can't just correct by surface_pts_upsample_factor because of how interpolation works...
    pt = [sf_pts[..., f] * (X_sdt.shape[f] - 1) / (X_edges.shape[f] - 1) for f in range(X_sdt.ndim)]
    return np.stack(pt, -1)


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def Get_Jac(disp):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (disp[:,1:,:-1,:-1,:] - disp[:,:-1,:-1,:-1,:])
    D_x = (disp[:,:-1,1:,:-1,:] - disp[:,:-1,:-1,:-1,:])
    D_z = (disp[:,:-1,:-1,1:,:] - disp[:,:-1,:-1,:-1,:])
 
    D1 = (D_x[...,0]+1) * ((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1]) * (D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2]) * (D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = D1 - D2 + D3
    
    return D


def Get_Ja(disp):
  
    D_y = (disp[:,1:,:-1,:-1,:] - disp[:,:-1,:-1,:-1,:])
    D_x = (disp[:,:-1,1:,:-1,:] - disp[:,:-1,:-1,:-1,:])
    D_z = (disp[:,:-1,:-1,1:,:] - disp[:,:-1,:-1,:-1,:])
  
    D1 = (D_x[...,0]+1) * ( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1]) * (D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2]) * (D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = np.abs(D1-D2+D3)
 
    return D

def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<=0)

def save_deformation_field(
    flow: torch.Tensor,
    save_dir: str = "./deformation_fields",
    batch_indices: list = None,
    slice_indices: list = None,
    arrow_step: int = 4,
    flow_scale: float = 1.0,
    flow_cmap: str = "viridis",
    save_format: str = "png",
    dpi: int = 300,
    save_individual: bool = True,
    overwrite: bool = False,
    save_numpy: bool = False,
    convert_to_uint8: bool = False,
    flow_channel_order: str = "dz_dy_dx"  # 与SpatialTransformer一致：dz(0), dy(1), dx(2)
) -> None:
    """
    修复y轴颠倒问题：
    1. 绘图origin统一为"upper"（匹配医学图像(0,0)在左上角）
    2. 确保flow_dy方向与医学图像y轴（从上到下）一致
    """
    # ---------------------- 1. 初始化配置 ----------------------
    # save_dir = Path(save_dir)
    # save_dir.mkdir(parents=True, exist_ok=overwrite)
    
    B = flow.shape[0]
    batch_indices = batch_indices if batch_indices is not None else list(range(B))
    batch_indices = [idx for idx in batch_indices if 0 <= idx < B]
    
    D = flow.shape[2]  # flow shape: B×3×D×H×W → 第2维是D
    slice_indices = slice_indices if slice_indices is not None else [max(0, D//2-1), D//2, min(D-1, D//2+1)]
    slice_indices = [idx for idx in slice_indices if 0 <= idx < D]
    
    # 图像/热力图处理（不翻转y轴，匹配医学图像）
    def process_img(img):
        img_min = img.min()
        img_max = img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        if convert_to_uint8:
            img_norm = (img_norm * 255).clip(0, 255).astype(np.uint8)
        return img_norm  # 不做垂直翻转，保持医学图像的y轴方向

    # ---------------------- 2. 批量处理 ----------------------
    for batch_idx in batch_indices:
        flow_batch = flow[batch_idx].detach().cpu().numpy()  # (3, D, H, W) → 通道：dz, dy, dx
        # batch_dir = save_dir / f"batch_{batch_idx:02d}"
        # batch_dir.mkdir(exist_ok=overwrite)
                
        # 保存原始形变场
        if save_numpy:
            np.save(batch_dir / f"flow_batch_{batch_idx:02d}.npy", flow_batch)
        
        for slice_idx in slice_indices:
            # 提取当前切片的形变场（通道：dz, dy, dx）
            flow_slice = flow_batch[slice_idx] # (H, W, 3)
            H, W = flow_slice.shape[:2]
                        
            # ---------------------- 3. 关键：形变场通道映射（与SpatialTransformer一致） ----------------------
            if flow_channel_order == "dz_dy_dx":
                flow_dz = flow_slice[..., 0]  # D轴位移（切片间，不影响2D可视化）
                flow_dy = flow_slice[..., 1]  # H轴（y轴）位移 → 医学图像y轴：从上到下为正
                flow_dx = flow_slice[..., 2]  # W轴（x轴）位移 → 医学图像x轴：从左到右为正
            else:
                raise ValueError("flow_channel_order必须为'dz_dy_dx'（与SpatialTransformer匹配）")
            
            # 位移大小（热力图，不受方向影响）
            flow_magnitude = np.sqrt(flow_dx**2 + flow_dy**2 + flow_dz**2)
            flow_mag_processed = process_img(flow_magnitude)
            
            # ---------------------- 4. 箭头网格采样（匹配origin="upper"） ----------------------
            # x: W轴（从左到右），y: H轴（从上到下）
            x = np.arange(0, W, arrow_step)
            y = np.arange(0, H, arrow_step)
            xx, yy = np.meshgrid(x, y, indexing="xy")  # 匹配Matplotlib的xy索引
            
            # 采样位移（flow_dy正方向=从上到下，与医学图像一致）
            dx_sampled = flow_dx[yy, xx] * flow_scale  # x轴位移：正=向右
            dy_sampled = flow_dy[yy, xx] * flow_scale  # y轴位移：正=向下（匹配医学图像y轴）
            
            # ---------------------- 5. 可视化（origin="upper"，核心修复！） ----------------------
            fig_cols = 2
            fig, axes = plt.subplots(1, fig_cols, figsize=(18 if fig_cols==3 else 12, 6))
            fig.suptitle(f"Deformation Field (Batch {batch_idx}, Slice {slice_idx})", fontsize=16)
            
            # 5.1 箭头图（origin="upper"）
            ax1 = axes[0]
            ax1.imshow(flow_mag_processed, cmap=flow_cmap, origin="upper", alpha=0.7)  # 关键：origin="upper"
            quiver = ax1.quiver(
                xx, yy, dx_sampled, dy_sampled,
                angles="xy", scale_units="xy", scale=1,
                color="red", alpha=0.8, linewidths=0.5
            )
            ax1.set_title("Displacement Vectors (Y: Top→Bottom)")
            ax1.set_xlabel("Width (W) →")
            ax1.set_ylabel("Height (H) →")  # 明确y轴方向：从上到下
            ax1.quiverkey(quiver, 0.9, 0.9, flow_scale, f"Scale: {flow_scale}", coordinates="axes")
            
            # 5.2 热力图（origin="upper"，与箭头图对齐）
            ax2 = axes[1]
            im2 = ax2.imshow(flow_mag_processed, cmap=flow_cmap, origin="upper")  # 关键：origin="upper"
            ax2.set_title("Displacement Magnitude")
            ax2.set_xlabel("Width (W) →")
            ax2.set_ylabel("Height (H) →")
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            plt.tight_layout()
            
            # ---------------------- 6. 保存图像 ----------------------
            # combo_path = save_dir / f"flow_combo_batch{batch_idx:02d}_slice{slice_idx:02d}.{save_format}"
            # if not combo_path.exists() or overwrite:
            #     plt.savefig(
            #         combo_path,
            #         dpi=dpi,
            #         bbox_inches="tight",
            #         pad_inches=0.2,
            #         format=save_format.lower()
            #     )
            plt.close(fig)
            
            # ---------------------- 7. 单独保存子图（可选） ----------------------
            if save_individual:
                # individual_dir = batch_dir / f"slice_{slice_idx:02d}"
                # individual_dir.mkdir(exist_ok=overwrite)
                
                # 箭头图
                fig_arrow, ax_arrow = plt.subplots(figsize=(8, 6))
                ax_arrow.imshow(flow_mag_processed, cmap=flow_cmap, origin="upper", alpha=0.7)
                ax_arrow.quiver(xx, yy, dx_sampled, dy_sampled, angles="xy", scale_units="xy", scale=1, color="red", alpha=0.8)
                # ax_arrow.set_title(f"Displacement Vectors (Batch {batch_idx}, Slice {slice_idx})")
                ax_arrow.set_xlabel("Width (W) →")
                ax_arrow.set_ylabel("Height (H) →")
                ax_arrow.quiverkey(quiver, 0.9, 0.9, flow_scale, f"Scale: {flow_scale}", coordinates="axes")
                ax_arrow.set_xticks([])  # 隐藏X轴刻度
                ax_arrow.set_yticks([])  # 隐藏Y轴刻度
                ax_arrow.set_xlabel('')   # 清空X轴标签
                ax_arrow.set_ylabel('')   # 清空Y轴标签
                ax_arrow.spines['top'].set_visible(False)    # 隐藏顶部边框
                ax_arrow.spines['right'].set_visible(False)  # 隐藏右侧边框
                ax_arrow.spines['bottom'].set_visible(False) # 隐藏底部边框
                ax_arrow.spines['left'].set_visible(False)   # 隐藏左侧边框
                ax_arrow.figure.savefig(save_dir / f"{slice_idx:02d}_flow_arrow.{save_format}", dpi=dpi, bbox_inches="tight")
                plt.close(fig_arrow)
                
                # 热力图
                fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
                im_heat = ax_heat.imshow(flow_mag_processed, cmap=flow_cmap, origin="upper")
                # ax_heat.set_title(f"Displacement Magnitude (Batch {batch_idx}, Slice {slice_idx})")
                # ax_heat.set_xlabel("Width (W) →")
                # ax_heat.set_ylabel("Height (H) →")
                plt.colorbar(im_heat, ax=ax_heat, shrink=0.8)
                ax_heat.figure.savefig(save_dir / f"{slice_idx:02d}_flow_heat.{save_format}", dpi=dpi, bbox_inches="tight")
                plt.close(fig_heat)
            
            # 保存单个切片的形变场
            if save_numpy:
                numpy_slice_path = save_dir / f"flow_slice{slice_idx:02d}.npy"
                np.save(numpy_slice_path, flow_slice)
        
        # print(f"Batch {batch_idx} 保存完成：{batch_dir}")
    
    # print(f"所有结果已保存到：{save_dir}")
    

def visualize_registration_error(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    save_dir: str = "./registration_error",
    batch_indices: list = None,
    slice_indices: list = None,
    error_metric: str = "mae",  # 误差指标："mae"（绝对误差）或 "mse"（平方误差）
    show_heat: bool = True,
    show_mae: bool = True,
    show_ssim: bool = True,  # 是否显示SSIM相似度图（1=完美配准，0=完全不匹配）
    show_ncc: bool = True,  # 是否显示NCC相似度图
    ncc_win_size: int = 7,  # 新增：NCC局部窗口大小（奇数）
    cmap_img: str = "gray",
    cmap_error: str = "hot",
    cmap_ssim: str = "viridis",
    cmap_ncc: str = "plasma",
    save_format: str = "png",
    dpi: int = 300,
    save_individual: bool = True,
    overwrite: bool = False,
    save_numpy: bool = False,  # 保存误差数值（.npy）用于定量分析
    convert_to_uint8: bool = True  # 图像转为uint8增强兼容性
) -> None:
    """
    可视化固定图像与配准后图像的配准误差
    
    参数说明：
        fixed: 固定图像张量 (B×C×D×H×W)，C=1（单通道医学图像）
        warped: 配准后图像张量 (B×C×D×H×W)，与fixed维度完全一致
        save_dir: 保存目录
        batch_indices: 待处理的batch索引（None=全部）
        slice_indices: 待处理的切片索引（None=中间3个切片）
        error_metric: 误差计算方式："mae"（绝对误差）或 "mse"（平方误差）
        show_ssim: 是否显示结构相似性（SSIM）图（配准越好，SSIM越接近1）
        cmap_img: 原始图像的colormap
        cmap_error: 误差热力图的colormap
        cmap_ssim: SSIM图的colormap
        其他参数：与形变场保存函数一致，保持使用习惯统一
    """
    # ---------------------- 1. 初始化配置（与之前函数保持一致） ----------------------
    # 校验输入维度一致性
    assert fixed.shape == warped.shape, f"fixed与warped维度不匹配！{fixed.shape} vs {warped.shape}"
    assert fixed.shape[1] == 1, "仅支持单通道（C=1）医学图像"
    
    # save_dir = Path(save_dir)
    # save_dir.mkdir(parents=True, exist_ok=overwrite)
    
    # 处理batch索引
    B = fixed.shape[0]
    batch_indices = batch_indices if batch_indices is not None else list(range(B))
    batch_indices = [idx for idx in batch_indices if 0 <= idx < B]
    
    # 处理slice索引
    D, H, W = fixed.shape[2:]
    slice_indices = slice_indices if slice_indices is not None else [max(0, D//2-1), D//2, min(D-1, D//2+1)]
    slice_indices = [idx for idx in slice_indices if 0 <= idx < D]
    
    # ---------------------- 2. 核心工具函数 ----------------------
    def process_image(img):
        """图像归一化+可选uint8转换（与形变场函数逻辑一致）"""
        img_min = img.min()
        img_max = img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        if convert_to_uint8:
            img_norm = (img_norm * 255).clip(0, 255).astype(np.uint8)
        return img_norm
    
    def calculate_error(fixed_slice, warped_slice):
        """计算配准误差（MAE/MSE）"""
        # 确保输入为float类型，避免整数溢出
        fixed_float = fixed_slice.astype(np.float32)
        warped_float = warped_slice.astype(np.float32)
        
        if error_metric == "mae":
            error = np.abs(fixed_float - warped_float)
        elif error_metric == "mse":
            error = (fixed_float - warped_float) ** 2
        else:
            raise ValueError(f"不支持的误差指标：{error_metric}，仅支持'mae'/'mse'")
        
        # 归一化误差到[0,1]（用于可视化）
        error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
        return error, error_norm
    
    def calculate_ssim(fixed_slice, warped_slice):
        """计算结构相似性（SSIM）：取值范围[-1,1]，归一化到[0,1]用于可视化"""
        # SSIM要求输入为float32，且范围[0,1]
        fixed_norm = process_image(fixed_slice) / 255.0 if convert_to_uint8 else process_image(fixed_slice)
        warped_norm = process_image(warped_slice) / 255.0 if convert_to_uint8 else process_image(warped_slice)
        
        # 计算SSIM（win_size需为奇数，且不超过图像最小维度）
        win_size = min(7, fixed_slice.shape[0]//2, fixed_slice.shape[1]//2)
        ssim_val, ssim_map = ssim(
            fixed_norm, warped_norm, full=True, win_size=win_size,
            data_range=1.0, channel_axis=None  # 单通道图像
        )
        # 归一化到[0,1]（原始SSIM可能为负，配准差时接近0）
        ssim_map_norm = (ssim_map + 1) / 2  # 转为[0,1]
        return ssim_val, ssim_map_norm
    
    def calculate_local_ncc(fixed_slice, warped_slice, win_size=7):
        """
        计算局部归一化互相关（NCC）：逐像素生成NCC热力图
        :param fixed_slice: 固定图像切片 (H×W)
        :param warped_slice: 配准后图像切片 (H×W)
        :param win_size: 局部窗口大小（奇数）
        :return: ncc_mean（全局均值）, ncc_map_norm（归一化到[0,1]的局部NCC图）
        """
        # 转为float32，避免数值溢出
        fixed = fixed_slice.astype(np.float32)
        warped = warped_slice.astype(np.float32)

        # 1. 计算窗口内的均值（卷积实现）
        kernel = np.ones((win_size, win_size)) / (win_size ** 2)
        fixed_mean = convolve(fixed, kernel, mode='reflect')  # 反射填充避免边界失真
        warped_mean = convolve(warped, kernel, mode='reflect')

        # 2. 中心化（减去窗口均值）
        fixed_centered = fixed - fixed_mean
        warped_centered = warped - warped_mean

        # 3. 计算NCC分子（协方差）和分母（标准差乘积）
        numerator = convolve(fixed_centered * warped_centered, kernel, mode='reflect')
        fixed_std = np.sqrt(convolve(fixed_centered ** 2, kernel, mode='reflect') + 1e-8)
        warped_std = np.sqrt(convolve(warped_centered ** 2, kernel, mode='reflect') + 1e-8)
        denominator = fixed_std * warped_std + 1e-8  # 避免除零

        # 4. 计算局部NCC（取值[-1,1]）
        ncc_map = numerator / denominator
        ncc_mean = np.mean(ncc_map)  # 全局均值（定量指标）

        # 5. 归一化到[0,1]用于可视化（0=最差，1=最优）
        ncc_map_norm = (ncc_map + 1) / 2

        return ncc_mean, ncc_map_norm
    
    # ---------------------- 3. 批量处理与可视化 ----------------------
    ncc_func = NCC()
    for batch_idx in batch_indices:
        # 提取单个batch的图像（转为numpy，取第0通道）
        fixed_batch = fixed[batch_idx, 0].detach().cpu().numpy()  # (D×H×W)
        warped_batch = warped[batch_idx, 0].detach().cpu().numpy()
        
        # 创建batch保存目录
        # batch_dir = save_dir / f"batch_{batch_idx:02d}"
        # batch_dir.mkdir(exist_ok=overwrite)
        
        for slice_idx in slice_indices:
            # 提取当前切片
            fixed_slice = fixed_batch[slice_idx]  # (H×W)
            warped_slice = warped_batch[slice_idx]
            H, W = fixed_slice.shape
            
            # 处理原始图像（归一化+可选uint8）
            fixed_processed = process_image(fixed_slice)
            warped_processed = process_image(warped_slice)
            
            # 计算误差（数值+归一化用于可视化）
            error, error_norm = calculate_error(fixed_slice, warped_slice)
            # 可选计算SSIM
            ssim_val, ssim_map_norm = calculate_ssim(fixed_slice, warped_slice) if show_ssim else (None, None)
            ncc_mean, ncc_map_norm = calculate_local_ncc(fixed_slice, warped_slice, win_size=ncc_win_size)
            ncc_map_norm = ncc_func.loss(torch.tensor(fixed_slice).reshape(1,1,H,W), torch.tensor(warped_slice).reshape(1,1,H,W), reduction='None')[0,0] if show_ncc else None
            ncc_map_norm = ncc_map_norm + 1
            # ---------------------- 4. 可视化布局 ----------------------
            # 列数：3列（固定图+配准后图+误差图）或4列（加SSIM图）
            n_cols = 3
            if show_ssim: n_cols += 1
            if show_ncc: n_cols += 1

            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))  # 自适应列宽
            fig.suptitle(
                f"Registration Error (Batch {batch_idx}, Slice {slice_idx}) | "
                f"Error Metric: {error_metric.upper()} | SSIM: {ssim_val:.3f}" if show_ssim else
                f"Registration Error (Batch {batch_idx}, Slice {slice_idx}) | Error Metric: {error_metric.upper()}",
                fontsize=16
            )
            
            # 4.1 固定图像
            ax1 = axes[0]
            ax1.imshow(fixed_processed, cmap=cmap_img, origin="upper")  # 匹配医学图像(0,0)在左上角
            ax1.set_title("Fixed Image")
            ax1.set_xlabel("Width (W) →")
            ax1.set_ylabel("Height (H) →")
            plt.colorbar(ax1.imshow(fixed_processed, cmap=cmap_img, origin="upper"), ax=ax1, shrink=0.8)
            
            # 4.2 配准后图像
            ax2 = axes[1]
            ax2.imshow(warped_processed, cmap=cmap_img, origin="upper")
            ax2.set_title("Warped Image (Registered)")
            ax2.set_xlabel("Width (W) →")
            ax2.set_ylabel("Height (H) →")
            plt.colorbar(ax2.imshow(warped_processed, cmap=cmap_img, origin="upper"), ax=ax2, shrink=0.8)
            
            # 4.3 误差热力图（核心）
            ax3 = axes[2]
            im_error = ax3.imshow(error_norm, cmap=cmap_error, origin="upper")
            ax3.set_title(f"{error_metric.upper()} Error (Max: {error.max():.2f})")
            ax3.set_xlabel("Width (W) →")
            ax3.set_ylabel("Height (H) →")
            cbar_error = plt.colorbar(im_error, ax=ax3, shrink=0.8)
            cbar_error.set_label("Normalized Error")
            
            # 4.4 可选：SSIM相似度图（配准越好，颜色越亮）
            ax_idx = 3
            if show_ssim:
                ax4 = axes[ax_idx]
                im_ssim = ax4.imshow(ssim_map_norm, cmap=cmap_ssim, origin="upper")
                # ax4.set_title(f"SSIM Map (Mean: {ssim_val:.3f})")
                # ax4.set_xlabel("Width (W) →")
                # ax4.set_ylabel("Height (H) →")
                cbar_ssim = plt.colorbar(im_ssim, ax=ax4, shrink=0.8)
                cbar_ssim.set_label("Normalized SSIM (0=Bad, 1=Perfect)")
                ax_idx = ax_idx + 1
            
            # 子图5：NCC图（新增，可选）
            if show_ncc:
                ax5 = axes[ax_idx]
                im_ncc = ax5.imshow(ncc_map_norm, cmap=cmap_ncc, origin="upper")
                # ax5.set_title(f"NCC Map (Mean: {ncc_mean:.3f}) | Win: {ncc_win_size}")
                # ax5.set_xlabel("Width (W) →")
                # ax5.set_ylabel("Height (H) →")
                cbar_ncc = plt.colorbar(im_ncc, ax=ax5, shrink=0.8)
                cbar_ncc.set_label("Normalized NCC (0=Bad, 1=Perfect)")
                
            plt.tight_layout()
            
            # ---------------------- 5. 保存图像 ----------------------
            # combo_filename = batch_dir / f"error_combo_batch{batch_idx:02d}_slice{slice_idx:02d}.{save_format}"
            # if not combo_filename.exists() or overwrite:
            #     plt.savefig(
            #         combo_filename,
            #         dpi=dpi,
            #         bbox_inches="tight",
            #         pad_inches=0.2,
            #         format=save_format.lower()
            #     )
            plt.close(fig)
            
            # ---------------------- 6. 单独保存子图（可选） ----------------------
            if save_individual:
                # individual_dir = batch_dir / f"slice_{slice_idx:02d}"
                # individual_dir.mkdir(exist_ok=overwrite)
                
                # 保存固定图像
                # fig_fixed, ax_fixed = plt.subplots(figsize=(8, 6))
                # ax_fixed.imshow(fixed_processed, cmap=cmap_img, origin="upper")
                # ax_fixed.set_title(f"Fixed Image (Batch {batch_idx}, Slice {slice_idx})")
                # plt.colorbar(ax_fixed.imshow(fixed_processed, cmap=cmap_img, origin="upper"), ax=ax_fixed)
                # fig_fixed.savefig(individual_dir / f"fixed_slice{slice_idx:02d}.{save_format}", dpi=dpi, bbox_inches="tight")
                # plt.close(fig_fixed)
                
                # 保存配准后图像
                # fig_warped, ax_warped = plt.subplots(figsize=(8, 6))
                # ax_warped.imshow(warped_processed, cmap=cmap_img, origin="upper")
                # ax_warped.set_title(f"Warped Image (Batch {batch_idx}, Slice {slice_idx})")
                # plt.colorbar(ax_warped.imshow(warped_processed, cmap=cmap_img, origin="upper"), ax=ax_warped)
                # fig_warped.savefig(individual_dir / f"warped_slice{slice_idx:02d}.{save_format}", dpi=dpi, bbox_inches="tight")
                # plt.close(fig_warped)
                
                # 保存误差热力图
                if show_heat:
                    fig_error, ax_error = plt.subplots(figsize=(8, 6))
                    im_err = ax_error.imshow(error_norm, cmap=cmap_error, origin="upper")
                    ax_error.set_title(f"{error_metric.upper()} Error (Batch {batch_idx}, Slice {slice_idx})")
                    plt.colorbar(im_err, ax=ax_error)
                    fig_error.savefig(save_dir / f"{slice_idx:02d}_error_heat_slice.{save_format}", dpi=dpi, bbox_inches="tight")
                    plt.close(fig_error)
                
                # 保存误差图
                if show_mae:
                    fig_error, ax_error = plt.subplots(figsize=(8, 6))
                    im_err = ax_error.imshow(error_norm[...,np.newaxis].repeat(3,-1), cmap=cmap_error, origin="upper")
                    # fig_error = Image.fromarray((error_norm*255).astype(np.uint8), mode='L')
                    fig_error.set_xticks([])  # 隐藏X轴刻度
                    fig_error.set_yticks([])  # 隐藏Y轴刻度
                    fig_error.set_xlabel('')   # 清空X轴标签
                    fig_error.set_ylabel('')   # 清空Y轴标签
                    fig_error.spines['top'].set_visible(False)    # 隐藏顶部边框
                    fig_error.spines['right'].set_visible(False)  # 隐藏右侧边框
                    fig_error.spines['bottom'].set_visible(False) # 隐藏底部边框
                    fig_error.spines['left'].set_visible(False)   # 隐藏左侧边框
                    fig_error.savefig(save_dir / f"{slice_idx:02d}_error_slice.{save_format}", dpi=dpi, bbox_inches="tight")
                    plt.close(fig_error)
                
                # 保存SSIM图（可选）
                if show_ssim:
                    fig_ssim, ax_ssim = plt.subplots(figsize=(8, 6))
                    im_ssim = ax_ssim.imshow(ssim_map_norm, cmap=cmap_ssim, origin="upper")
                    # ax_ssim.set_title(f"SSIM Map (Batch {batch_idx}, Slice {slice_idx}) | Mean: {ssim_val:.3f}")
                    plt.colorbar(im_ssim, ax=ax_ssim)
                    fig_ssim.savefig(save_dir / f"{slice_idx:02d}_ssim_slice.{save_format}", dpi=dpi, bbox_inches="tight")
                    plt.close(fig_ssim)
                
                if show_ncc:
                    fig_ncc, ax_ncc = plt.subplots(figsize=(8, 6))
                    im_ncc = ax_ncc.imshow(ncc_map_norm, cmap=cmap_ssim, origin="upper")
                    # ax_ncc.set_title(f"NCC Map (Batch {batch_idx}, Slice {slice_idx}) | Mean: {ncc_mean:.3f}")
                    plt.colorbar(im_ncc, ax=ax_ncc)
                    fig_ncc.savefig(save_dir / f"{slice_idx:02d}_ncc_slice.{save_format}", dpi=dpi, bbox_inches="tight")
                    plt.close(fig_ncc)
                
    # print(f"所有配准误差结果已保存到：{save_dir}")

def visualize_registration_results(
    flow: torch.Tensor,
    fixed: torch.Tensor,
    warped: torch.Tensor,
    save_dir: str = "./registration_combined",
    batch_indices: list = None,
    slice_indices: list = None,
    # 形变场可视化参数
    flow_channel_order: str = "dz_dy_dx",
    arrow_step: int = 8,
    flow_scale: float = 1.0,
    cmap_flow_mag: str = "viridis",
    # 误差计算参数
    error_metric: str = "mae",
    show_mae: bool = True,
    show_ssim: bool = True,
    show_ncc: bool = True,
    ncc_win_size: int = 7,
    cmap_error: str = "hot",
    cmap_ssim: str = "viridis",
    cmap_ncc: str = "plasma",
    # 通用保存参数
    save_format: str = "png",
    dpi: int = 150,
    overwrite: bool = False,
    save_numpy: bool = False,
    convert_to_uint8: bool = True
) -> None:
    """
    合并形变场可视化 + 配准误差计算，仅单独保存各类指标图像（无组合图、无原图）
    保存内容：
    - 形变场：位移向量图（箭头）、位移大小热力图（单通道）
    - 配准误差：MAE/MSE误差图（单通道）、SSIM图（单通道）、NCC图（单通道）
    所有图像单独保存，无弹窗，支持批量处理
    """
    # 输入校验
    assert flow.ndim == 5 and fixed.ndim == 5 and warped.ndim == 5, "输入必须为B×C×D×H×W张量"
    assert flow.shape[0] == fixed.shape[0] == warped.shape[0], "Batch维度不匹配"
    assert fixed.shape[1] == 1 and warped.shape[1] == 1, "仅支持单通道（C=1）医学图像"
    assert flow_channel_order in ["dz_dy_dx"], "仅支持dz_dy_dx通道顺序（匹配SpatialTransformer）"
    assert error_metric in ["mae", "mse"], "误差指标仅支持mae/mse"
    assert ncc_win_size % 2 == 1, "NCC窗口大小必须为奇数"
    assert save_format.lower() in ["png", "jpg", "jpeg"], "仅支持png/jpg格式"

    # 初始化保存目录
    # save_dir = Path(save_dir)
    # save_dir.mkdir(parents=True, exist_ok=overwrite)

    # 处理batch/slice索引
    B = flow.shape[0]
    batch_indices = batch_indices if batch_indices is not None else list(range(B))
    batch_indices = [idx for idx in batch_indices if 0 <= idx < B]

    D = flow.shape[2]  # flow: B×3×D×H×W | fixed: B×1×D×H×W
    slice_indices = slice_indices if slice_indices is not None else [max(0, D//2-1), D//2, min(D-1, D//2+1)]
    slice_indices = [idx for idx in slice_indices if 0 <= idx < D]

    # ---------------------- 工具函数 ----------------------
    def process_array(arr):
        """数组归一化到[0,1]，可选转为uint8"""
        arr_min = arr.min()
        arr_max = arr.max()
        arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
        if convert_to_uint8:
            arr_norm = (arr_norm * 255).clip(0, 255).astype(np.uint8)
        return arr_norm

    def save_single_channel_img(arr, save_path, origin_upper=True):
        """保存纯单通道图像（无画布/坐标轴）"""
        if arr.dtype != np.uint8:
            arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr_uint8 = arr
        if not origin_upper:
            arr_uint8 = np.flipud(arr_uint8)
        img = Image.fromarray(arr_uint8, mode='L')
        save_kwargs = {"quality": 95} if save_path.suffix.lower() in [".jpg", ".jpeg"] else {}
        img.save(save_path, **save_kwargs)

    def process_flow(flow_slice):
        """处理形变场切片，返回位移大小和dx/dy（用于箭头图）"""
        # flow_slice: (H×W×3)，通道顺序dz(0), dy(1), dx(2)
        flow_dz = flow_slice[..., 0]
        flow_dy = flow_slice[..., 1]  # H轴（y）位移，origin="upper"时正方向向下
        flow_dx = flow_slice[..., 2]  # W轴（x）位移，正方向向右
        
        # 计算位移大小（L2范数）
        flow_magnitude = np.sqrt(flow_dx**2 + flow_dy**2 + flow_dz**2)
        flow_magnitude_norm = process_array(flow_magnitude)
        
        return flow_magnitude, flow_magnitude_norm, flow_dx, flow_dy

    def calculate_error(fixed_slice, warped_slice):
        """计算MAE/MSE误差"""
        fixed_float = fixed_slice.astype(np.float32)
        warped_float = warped_slice.astype(np.float32)
        if error_metric == "mae":
            error = np.abs(fixed_float - warped_float)
        else:
            error = (fixed_float - warped_float) ** 2
        error_norm = process_array(error)
        return error, error_norm

    def calculate_ssim(fixed_slice, warped_slice):
        """计算SSIM"""
        fixed_norm = process_array(fixed_slice) / 255.0
        warped_norm = process_array(warped_slice) / 255.0
        win_size = min(7, fixed_slice.shape[0]//2, fixed_slice.shape[1]//2)
        ssim_val, ssim_map = ssim(
            fixed_norm, warped_norm, full=True, win_size=win_size,
            data_range=1.0, channel_axis=None
        )
        ssim_map_norm = (ssim_map + 1) / 2  # 归一化到[0,1]
        return ssim_val, ssim_map_norm

    def calculate_local_ncc(fixed_slice, warped_slice, win_size=7):
        """计算局部NCC"""
        fixed = fixed_slice.astype(np.float32)
        warped = warped_slice.astype(np.float32)
        # 窗口均值卷积
        kernel = np.ones((win_size, win_size)) / (win_size ** 2)
        fixed_mean = convolve(fixed, kernel, mode='reflect')
        warped_mean = convolve(warped, kernel, mode='reflect')
        # 中心化
        fixed_centered = fixed - fixed_mean
        warped_centered = warped - warped_mean
        # 计算NCC
        numerator = convolve(fixed_centered * warped_centered, kernel, mode='reflect')
        fixed_std = np.sqrt(convolve(fixed_centered ** 2, kernel, mode='reflect') + 1e-8)
        warped_std = np.sqrt(convolve(warped_centered ** 2, kernel, mode='reflect') + 1e-8)
        denominator = fixed_std * warped_std + 1e-8
        ncc_map = numerator / denominator
        ncc_mean = np.mean(ncc_map)
        ncc_map_norm = (ncc_map + 1) / 2  # 归一化到[0,1]
        return ncc_mean, ncc_map_norm

    # ---------------------- 批量处理主逻辑 ----------------------
    for batch_idx in batch_indices:
        # 提取当前batch数据
        flow_batch = flow[batch_idx].detach().cpu().numpy()  # (3×D×H×W)
        fixed_batch = fixed[batch_idx, 0].detach().cpu().numpy()  # (D×H×W)
        warped_batch = warped[batch_idx, 0].detach().cpu().numpy()  # (D×H×W)
        
        # batch_dir = save_dir / f"batch_{batch_idx:02d}"
        # batch_dir.mkdir(exist_ok=overwrite)

        for slice_idx in slice_indices:
            # 创建切片级保存目录
            # slice_dir = batch_dir / f"slice_{slice_idx:02d}"
            # slice_dir.mkdir(exist_ok=overwrite)

            # ---------------------- 1. 形变场处理与保存 ----------------------
            # 提取并处理形变场切片
            flow_slice = flow_batch[:, slice_idx, :, :].transpose(1, 2, 0)  # (3×H×W) → (H×W×3)
            flow_magnitude, flow_magnitude_norm, flow_dx, flow_dy = process_flow(flow_slice)
            H, W = flow_magnitude.shape

            # 1.1 保存形变场位移大小热力图（单通道）
            flow_mag_path = save_dir / f"{slice_idx:02d}_flow_magnitude.{save_format.lower()}"
            save_single_channel_img(flow_magnitude_norm, flow_mag_path)

            # 1.2 保存形变场箭头图（单独文件，无弹窗）
            flow_arrow_path = save_dir / f"{slice_idx:02d}_flow_vectors.{save_format.lower()}"
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(flow_magnitude_norm, cmap=cmap_flow_mag, origin="upper")
            # 采样箭头（避免密集）
            x = np.arange(0, W, arrow_step)
            y = np.arange(0, H, arrow_step)
            xx, yy = np.meshgrid(x, y)
            dx_sampled = flow_dx[yy, xx] * flow_scale
            dy_sampled = flow_dy[yy, xx] * flow_scale
            ax.quiver(
                xx, yy, dx_sampled, dy_sampled,
                angles="xy", scale_units="xy", scale=1,
                color="red", alpha=0.8, linewidths=0.5
            )
            ax.set_title(f"Flow Vectors (Slice {slice_idx})")
            ax.set_xlabel("Width (W) →")
            ax.set_ylabel("Height (H) →")
            fig.savefig(flow_arrow_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            # ---------------------- 2. 配准误差处理与保存 ----------------------
            # 提取固定/配准后图像切片
            fixed_slice = fixed_batch[slice_idx]
            warped_slice = warped_batch[slice_idx]

            # 2.1 计算并保存MAE/MSE误差图（单通道）
            if show_mae:
                error, error_norm = calculate_error(fixed_slice, warped_slice)
                error_path = save_dir / f"{slice_idx:02d}_{error_metric}_error.{save_format.lower()}"
                save_single_channel_img(error_norm, error_path)

            # 2.2 计算并保存SSIM图（单通道，可选）
            if show_ssim:
                ssim_val, ssim_map_norm = calculate_ssim(fixed_slice, warped_slice)
                ssim_path = save_dir / f"{slice_idx:02d}_ssim.{save_format.lower()}"
                save_single_channel_img(ssim_map_norm, ssim_path)

            # 2.3 计算并保存NCC图（单通道，可选）
            if show_ncc:
                ncc_mean, ncc_map_norm = calculate_local_ncc(fixed_slice, warped_slice, win_size=ncc_win_size)
                ncc_path = save_dir / f"{slice_idx:02d}_ncc.{save_format.lower()}"
                save_single_channel_img(ncc_map_norm, ncc_path)

            # ---------------------- 3. 保存数值文件（可选） ----------------------
            if save_numpy:
                # 保存形变场数值
                np.save(save_dir / "flow_magnitude.npy", flow_magnitude)
                np.save(save_dir / "flow_slice.npy", flow_slice)
                # 保存误差数值
                np.save(save_dir / f"{error_metric}_error.npy", error)
                # 保存SSIM数值
                if show_ssim:
                    np.save(save_dir / "ssim_map.npy", ssim_map_norm)
                    with open(save_dir / "ssim_mean.txt", "w") as f:
                        f.write(f"{ssim_val:.4f}")
                # 保存NCC数值
                if show_ncc:
                    np.save(save_dir / "ncc_map.npy", ncc_map_norm)
                    with open(save_dir / "ncc_mean.txt", "w") as f:
                        f.write(f"{ncc_mean:.4f}")

class FlowShow():
    def __init__(self, shape, grid_path, device):
        z, h, w = shape
        grid_pic = cv2.imread(grid_path,2)[np.newaxis,np.newaxis,...] # 1, 1, h, w
        self.grid_pic = torch.from_numpy(grid_pic).repeat(1,z,1,1).float().to(device)

    def show(self, stn, flow):
        b = flow.size(0)        
        return stn(self.grid_pic.unsqueeze(0).repeat(b,1,1,1,1), flow)

def create_grid(out_path, size=(128,128)):
    num1, num2 = (size[0]+10) // 10, (size[1]+10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0]) / 100.0, (size[1]) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(out_path)  # 保存图像
    # plt.show()

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def getMI(im1,im2):

    #im1 = im1.astype('float')
    #im2 = im2.astype('float')

    hang, lie = im1.shape
    count = hang*lie
    N = 256

    h = np.zeros((N,N))

    for i in range(hang):
        for j in range(lie):
            h[im1[i,j],im2[i,j]] = h[im1[i,j],im2[i,j]]+1

    h = h/np.sum(h)

    im1_marg = np.sum(h,axis=0)
    im2_marg = np.sum(h, axis=1)

    H_x = 0
    H_y = 0

    for i in range(N):
        if(im1_marg[i]!=0):
            H_x = H_x + im1_marg[i]*math.log2(im1_marg[i])

    for i in range(N):
        if(im2_marg[i]!=0):
            H_x = H_x + im2_marg[i]*math.log2(im2_marg[i])

    H_xy = 0

    for i in range(N):
        for j in range(N):
            if(h[i,j]!=0):
                H_xy = H_xy + h[i,j]*math.log2(h[i,j])

    MI = H_xy-H_x-H_y

    return MI

def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


if __name__ == '__main__':
    create_grid(out_path='grid_pic.jpg')