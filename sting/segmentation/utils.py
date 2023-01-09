# Some of the code is from the Omnipose repository
# https://github.com/kevinjohncutler/omnipose
#
import numpy as np
import scipy
import ncolor
import edt
import torch
import torch.optim as optim
import fastremap
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from skimage import io, filters
from skimage.measure import label, regionprops
from skimage.filters import gaussian

from scipy.ndimage import mean, convolve, binary_fill_holes, find_objects
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import remove_small_holes, remove_small_objects
from numba import njit, float32, int32, vectorize
from sklearn.cluster import DBSCAN

# faster DBSCAN should be imported here
# https://github.com/karempudi/dbscan-python 
# forked from a repo of paper https://dl.acm.org/doi/10.1145/3318464.3380582
# is used to compile this .so file, need to work on how to get it to work on
# windows
if os.name == 'posix':
    from .DBSCAN import DBSCAN as FASTDBSCAN
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

np.seterr(all='ignore')

import tqdm


def normalize_field(mu):
    mag = np.sqrt(np.nansum(mu**2,axis=0))
    m = mag>0
    mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag!=0,~np.isnan(mag)))        
    return mu

def normalize99(Y,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    X = Y.copy()
    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))


def diffuse_particles_omni_cpu(T, h_coordinates, w_coordinates, cell_w, niter):
    for t in range(niter):
        T[h_coordinates*cell_w + w_coordinates] = eikonal_update_cpu(T, h_coordinates, w_coordinates, cell_w)
        
    return T

def eikonal_update_cpu(T, h_coordinates, w_coordinates, cell_w):
    """Update for iterative solution to eikonal equation on CPU"""
    minx = np.minimum(T[h_coordinates*cell_w + w_coordinates - 1], T[h_coordinates*cell_w + w_coordinates + 1])
    miny = np.minimum(T[(h_coordinates - 1)*cell_w + w_coordinates], T[(h_coordinates + 1)*cell_w + w_coordinates])
    mina = np.minimum(T[(h_coordinates - 1)*cell_w + w_coordinates - 1], T[(h_coordinates + 1)*cell_w + w_coordinates + 1])
    minb = np.minimum(T[(h_coordinates - 1)*cell_w + w_coordinates + 1], T[(h_coordinates + 1)*cell_w + w_coordinates - 1])
    
        
    A = np.where(np.abs(mina-minb) >= 2, np.minimum(mina,minb)+np.sqrt(2), (1./2)*(mina+minb+np.sqrt(4-(mina-minb)**2)))
    B = np.where(np.abs(miny-minx) >= np.sqrt(2), np.minimum(miny,minx)+1, (1./2)*(miny+minx+np.sqrt(2-(miny-minx)**2)))
    
    return np.sqrt(A*B)


def labels_to_flows_cpu_omni(label_img):

    """
    Take a label_img image, with 0 background and cells = 1, 2,... N 
    and generate flows using the Omnipose method, i.e,
    
    Convert mask to flows using diffusion from center pixel
    
    Arguments:
        mask - numpy array (H, W), type: uint16
        
    Returns:
        label_img --
        dists --
        flows - flows in Y in mu[0] and flows in X in mu[1]
        heat - the field function whose derivatives are the flows
    """

    label_img = ncolor.format_labels(label_img)
    dists = edt.edt(label_img)

    img_pad = 15
    H_original, W_original = label_img.shape
    label_img_padded = np.pad(label_img, img_pad, mode='reflect')
    dists_padded = np.pad(dists, img_pad, mode='reflect')

    H_padded, W_padded = label_img_padded.shape
    flows_padded = np.zeros((2, H_padded, W_padded), np.float64)
    heat_padded = np.zeros((H_padded, W_padded), np.float64)

    n_cells = label_img_padded.max()

    slices = scipy.ndimage.find_objects(label_img_padded)

    cell_pad = 1

    # iterations to do on one image, calculate this for each image or cell.. not clear yet.. 
    # TODO: FIXIT 
    n_iter = 40

    # loop through the cells and calculate the field and flow and put it in 
    # mu_padded, mu_c_padded 
    for i, si in enumerate(slices):
        if si is not None:
            #print(f"Cell number: {i}")
            sr, sc = si

            one_cell = np.pad((label_img_padded[sr, sc] == i+1), cell_pad)

            cell_h, cell_w = one_cell.shape
            # find all non zero coordinates of each cell image that is padded
            h_coordinates, w_coordinates = np.nonzero(one_cell)
            h_coordinates, w_coordinates = h_coordinates.astype(np.int32), w_coordinates.astype(np.int32)

            #print(f"Cell H: {cell_h}, Cell W: {cell_w}")
            # heat that is arrayed in a line, instead of a an array. It will be reshaped later
            T = np.zeros(cell_h * cell_w, np.float64)
            
            # call the diffusion function to calculate the heat
            T = diffuse_particles_omni_cpu(T, h_coordinates, w_coordinates, cell_w, n_iter)

            # gradient of the heat along axis 0 (rows) and axis 1 (columns)
            dy = (T[(h_coordinates + 1) * cell_w + w_coordinates] 
                -T[(h_coordinates - 1) * cell_w + w_coordinates])/2
            dx = (T[h_coordinates * cell_w + w_coordinates + 1] 
                -T[h_coordinates * cell_w + w_coordinates -1])/2

            # place the caluclated values in the larger array using 
            # slice indices of each cell
            flows_padded[:, sr.start + h_coordinates - cell_pad, sc.start + w_coordinates - cell_pad] = np.stack((dy, dx))
            heat_padded[sr.start + h_coordinates - cell_pad, sc.start + w_coordinates - cell_pad] = T[h_coordinates* cell_w + w_coordinates]
    
    # normalize the flows
    flows_padded_normalized = normalize_field(flows_padded)

    # now we just remove the padding added around the image and return them
    return (label_img, 
            dists,
            heat_padded[img_pad: -img_pad, img_pad: -img_pad],
            flows_padded_normalized[:, img_pad: -img_pad, img_pad: -img_pad]
            )

def labels_to_output_omni(label_img):
    """
    Takes ina labelled image (H, W) and generates (8, H, W) that is used 
    by the network to calculcate the loss functions on

    Args:
        label_img (np.ndarray): a labeled image loaded using skimage.io
            of (H, W)

    Return
        labels (np.ndarray) : a set of images (8, H, W) stacked in the 
            following order. Look for labels_to_flows_cpu_omni documentation for 
            exact details to follow on these fields
                labels, dists, flow-x, flow-y, heat, boundary, binary_mask,
                weights
    """
    label_img, dists, heat, flows = labels_to_flows_cpu_omni(label_img)

    # final_labels (5, H, W) shape
    final_labels = np.concatenate((label_img[np.newaxis,:, :],
                                   dists[np.newaxis, :, :],
                                   flows,
                                   heat[np.newaxis, :, :]), axis=0).astype(np.float32)
    
    dist_bg = 5
    dist_t = final_labels[1]
    dist_t[dist_t == 0] = -5.0

    boundary = 5.0 * (final_labels[1] == 1)
    boundary[boundary == 0] = -5.0

    # add boundary to the final_labels stack
    final_labels = np.concatenate((final_labels, boundary[np.newaxis, ]))
    #add binary mask to the label stack
    binary_mask = final_labels[0] > 0
    final_labels = np.concatenate((final_labels, binary_mask[np.newaxis,]))

    # add weights

    bg_edt = edt.edt(final_labels[0] < 0, black_border=True)
    cutoff = 9
    weights = (gaussian(1 - np.clip(bg_edt, 0, cutoff)/ cutoff, 1) + 0.5)

    labels = np.concatenate((final_labels, weights[np.newaxis,]))

    return labels

def divergence_rescale(dP,mask):
    dP = dP.copy()
    dP *= mask 
    dP = normalize_field(dP)

    # compute the divergence
    Y, X = np.nonzero(mask)
    Ly,Lx = mask.shape
    pad = 1
    Tx = np.zeros((Ly+2*pad)*(Lx+2*pad), np.float64)
    Tx[Y*Lx+X] = np.reshape(dP[1].copy(),Ly*Lx)[Y*Lx+X]
    Ty = np.zeros((Ly+2*pad)*(Lx+2*pad), np.float64)
    Ty[Y*Lx+X] = np.reshape(dP[0].copy(),Ly*Lx)[Y*Lx+X]

    # Rescaling by the divergence
    div = np.zeros(Ly*Lx, np.float64)
    div[Y*Lx+X]=(Ty[(Y+2)*Lx+X]+8*Ty[(Y+1)*Lx+X]-8*Ty[(Y-1)*Lx+X]-Ty[(Y-2)*Lx+X]+
                 Tx[Y*Lx+X+2]+8*Tx[Y*Lx+X+1]-8*Tx[Y*Lx+X-1]-Tx[Y*Lx+X-2])
    div = normalize99(div)
    div.shape = (Ly,Lx)
    #add sigmoid on boundary output to help push pixels away - the final bit needed in some cases!
    # specifically, places where adjacent cell flows are too colinear and therefore had low divergence
    #mag = div+1/(1+np.exp(-bd))
    dP *= div
    return dP

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            step = dP[:,p0,p1]
            #step /= step_factor(t)
            step /= (1+t)
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
    return p

@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )

def steps2D_interp(p, dP, niter):
    shape = dP.shape[1:]
    dPt = np.zeros(p.shape, np.float32)
    for t in range(niter):
        #print(f"Iteration: {t}")
        map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
        #dPt /= step_factor(t)
        dPt /= (1 + t)
        for k in range(len(p)):
            p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
    return p

def steps2D_interp_gpu(p, dP, niter, device):
    shape = dP.shape[1:]

    shape = np.array(shape)[[1, 0]].astype('double') - 1

    pt = torch.from_numpy(p[[1,0]].T).double().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
    im = torch.from_numpy(dP[[1,0]]).double().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        # normalize pt between  0 and  1, normalize the flow
    for k in range(2): 
        im[:,k,:,:] *= 2./shape[k]
        pt[:,:,:,k] /= shape[k]
        
    # normalize to between -1 and 1
    pt = pt*2-1 
    
    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        dPt /= (1 + t)

        for k in range(2):
            pt[:, :, :, k] = torch.clamp(pt[:, :, :, k] + dPt[:, k, :, :], -1., 1.)


    pt = (pt + 1) * 0.5
    for k in range(2):
        pt[:, :, :, k] *= shape[k]

    p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T

    return p

# forget the traces, it is too expensive to copy for large images
def follow_flows_cpu_omni(dP, mask, niter=200, interp=True, use_gpu=False):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    
    # mesh grid to update the value of each pixel to the number (id) of the cell to 
    # which the pixel belongs to
    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)
    
    # basically row, column indices for pixels in mask and flow Y greater than certain value
    inds = np.array(np.nonzero(np.logical_or(mask,np.abs(dP[0])>1e-3))).astype(np.int32).T
    
    # check if there is only a few pixels, or indices are not found, probably pixel mask was empty
    if inds.ndim < 2 or inds.shape[0] < 5:
        # cutting out none to match arguments incase there are no cells
        return p, inds
    
    if not interp:
        p = steps2D(p, dP.astype(np.float32), inds, niter)
    else:
        p_interp = steps2D_interp(p[:, inds[:, 0], inds[:, 1]], dP, niter)
        p[:, inds[:, 0], inds[:, 1]] = p_interp
        
    return p, inds

# forget the traces, it is too expensive to copy for large images
def follow_flows_gpu_omni(dP, mask, niter=200, interp=True, device="cuda:0"):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    # mesh grid to update the value of each pixel to the number (id) of the cell to 
    # which the pixel belongs to
    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)
    
    # basically row, column indices for pixels in mask and flow Y greater than certain value
    inds = np.array(np.nonzero(np.logical_or(mask,np.abs(dP[0])>1e-3))).astype(np.int32).T
    
    # check if there is only a few pixels, or indices are not found, probably pixel mask was empty
    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, inds
     
    if not interp:
        p = steps2D(p, dP.astype(np.float32), inds, niter)
    else:
        p_interp = steps2D_interp_gpu(p[:, inds[:, 0], inds[:, 1]], dP, niter, device=device)
        p[:, inds[:, 0], inds[:, 1]] = p_interp
        
    return p, inds

def flow_error(mask, flows):
    """
    Calculate flow errors between the one generated by the net and the
    one generated using the flows empircially calculated from predicted
    masks.
   
    Arguments:
    ----------
    mask: (H, W) labelled array of cell masks,
          0=background, 1,2,3 ... cell mask numbers
    
    flows: (2, H, W) flow-y and flow-x outputs from the network
    
    Returns:
    --------
    flow_errors: float array with length mask.max() 
                For each mask calculate the mean squared error
                between net-predicted flows and flows calculated from masks.
    
    flows_calculated: (2, H, W) flow masks calculated on the masks outputed from the
                 reconstruction of the network output
    
    """ 
    
    assert flows.shape[1:] == mask.shape, f"Mask: {mask.shape} and flows: {flows.shape} shapes don't match"
    
    # ensure unique masks
    mask_i = np.reshape(np.unique(mask.astype(np.float32), return_inverse=True)[1], mask.shape)
    
    # remember flows_calculated will be in range [-1.0, 1.0]
    # but the flows predicted by net will be in range [-5.0, 5.0]
    _, _, _, flows_calculated = labels_to_flows_cpu_omni(mask_i)
    
    flow_errors = np.zeros(mask_i.max())
    for i in range(flows_calculated.shape[0]):
        
        # this is from scipy.ndimage mean, will calculate error over different
        # labelled regions
        flow_errors += mean((flows_calculated - flows/5.0)**2, mask_i, 
                           index=np.arange(1, mask_i.max() + 1))
        #print(flow_errors)
    return flow_errors, flows_calculated


def remove_bad_flows_masks(mask, flows, flow_threshold=0.4):
    """ 
    Remove masks which have inconsistent flows
    
    Compute flows from predicted masks and compare flows to predicted flows 
    from the network. Discards masks with flow errors greater than the 
    threshold.
    
    Arguments:
    ---------
    mask: (H, W) labelled array of cell masks,
          0=background, 1,2,3, ... cell mask numbers
    flows: (2, H, W) flow-y and flow-x outputs from the network
    
    threshold: float (default=0.4)
        masks with flow error greater than threshold are discarded.
        
    Returns:
    --------
    mask: (H, W) labelled array of cell masks,
          after removing bad/inconsistent masks
          
    """
    
    # mask_errors has errors calculated for each cell mask in the mask.
    # and arrayed according to their index
    mask_errors, _ = flow_error(mask, flows) 
    # get the indices that have error greater than the threshold
    bad_indices = 1 + (mask_errors > flow_threshold).nonzero()[0]
    # set bad indices to background
    mask[np.isin(mask, bad_indices)] = 0
    return mask

def fill_holes_and_remove_small_masks(mask, min_size=15, hole_size=3):
    """
    Fill holes in masks and discard masks smaller than min_size
    
    Arguments:
    ----------
    mask: (H, W) labelled array of cell masks,
          0=background, 1,2,3, ... cell mask numbers
    
    min_size: (int) default 15,
              minimum number of pixels per mask, can turn off with -1
    
    hole_size: size of the hole as a percent (defualt 3 percent)
    
    Returns:
    --------
    mask: (H, W) labelled array of cell mask, that are corrected by removing 
          smaller cell masks and also filling the holes,
          0=background, 1,2,3 ... cell mask numbers
    """
    mask = ncolor.format_labels(mask, min_area=min_size, clean=True)
    
    # slice objects and loop over each object and fill it for holes
    slices = find_objects(mask)
    j = 0
    for i, cell_slice in enumerate(slices):
        if cell_slice is not None:
            cell_mask = mask[cell_slice] == (i + 1)
            number_pixels = cell_mask.sum()
            
            # remove mask smaller than min_size
            if min_size > 0 and number_pixels < min_size:
                mask[cell_slice][cell_mask] = 0
            
            # other wise fill holes
            else:
                hole_size_percent = np.count_nonzero(cell_mask) * hole_size/100
                
                pad_mask = remove_small_holes(np.pad(cell_mask, 1, mode='constant'), hole_size_percent)
                cell_mask = pad_mask[1:-1, 1: -1]
            
            mask[cell_slice][cell_mask] = (j + 1)
            j += 1
    
    return mask
    

def reconstruct_masks_cpu_omni(net_output, cell_prob_threshold=0.3, clean_mask=True,
                        flow_threshold=0.4, min_size=15, hole_size=3, device="cpu", fast=False):
    """
    Reconstruct and return masks
    Arguments:
    ----------
    net_output: (4, H, W) np.ndarray consisting of the following arrays at indices 0:2, 2, 3 resp.
        dP: (2, H, W) np.ndarray , flow-y and flow-x each (H, W) in size
        dist: (H, W) distance image predicted by the network
        boundary: (H, W) distance 
    
    cell_prob_threshold: (float, default: 0.3) , threshold used on the distance field,
                         to get cell probability binary mask
    clean_mask: (bool, default: True) if True cleans the mask and removes bad mask with
                flows errors greater than flow_threshold (default: 0.4).
                clean_mask also removes smaller masks and fill holes
    
    flow_threshold: (float, default: 0.4), threshold used on the mean-squared error between
                calculated flows from masks generated by the network output reconstruction
                and flows predicted by the network.
    
    min_size: (int, default: 15), minimum size of the cell mask in pixels
    hole_size: (int, default: 3), holes of size in percent that are filled 
    Returns:
    --------
    masks : (H, W) labelled array of cell masks,
            0=background, 1,2,3 ... cell mask numbers
    """

    # if tensors are on GPU move them to CPU
    #if net_output.device.type == "cuda":
    #    dP = net_output.cpu().numpy()

    dP = net_output[:2]
    dist = net_output[2]
    boundary = net_output[3]

    # calculate the binary threshold
    cp_mask = dist > cell_prob_threshold
    # scale the divergence 
    dP_scaled = divergence_rescale(dP, cp_mask)

    if device == "cpu":
    # iterate and trace the flows back to origin points
        traced_p, inds = follow_flows_cpu_omni(dP_scaled, cp_mask, interp=False)
    elif device[:4] == "cuda":
        traced_p, inds = follow_flows_gpu_omni(dP_scaled, cp_mask, device=device)

    # distance field is used to generate the cell mask using the
    # cell_prob_threshold, this distance and diameter fields are 
    # not used new.
    #dist_cell_probs = np.abs(cell_probability[cp_mask])
    #d = 6 * np.mean(dist_cell_probs)
    eps = 1 + 1/3

    #y, x = np.nonzero(cp_mask)
    newinds = traced_p[:, inds[:, 0], inds[:, 1]].swapaxes(0, 1)

    new_mask = np.zeros((traced_p.shape[1], traced_p.shape[2]))

    if fast == False or os.name !='posix':
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=8).fit(newinds)
        labels = db.labels_
    elif (fast == True) and (os.name == 'posix') :
        labels, _ = FASTDBSCAN(newinds.astype('double'), eps=eps, min_samples=3)


    new_mask[inds[:, 0], inds[:, 1]] = labels + 1

    # clean the mask for bad flows and holes, and remove smaller masks
    if clean_mask and new_mask.max() > 0:

        # remove bad flows by comparing flow errors and removing large flow errors
        new_mask = remove_bad_flows_masks(new_mask, dP, flow_threshold=flow_threshold)
        _, new_mask = np.unique(new_mask, return_inverse=True)
        new_mask = np.reshape(new_mask, dP.shape[1:]).astype(np.int32)
        # fill small holes and remove masks smaller than min_size pixels
        mask_corrected = fill_holes_and_remove_small_masks(new_mask, min_size=min_size, hole_size=hole_size)
        # remap the indices, just incase they were altered above
        fastremap.renumber(mask_corrected, in_place=True)

        return mask_corrected

    else:
        # do a quick clean, forget removing bad flows and other things done above,
        # only remove small objects, it is good enough for our purpose
        new_mask = remove_small_objects(new_mask.astype('int'), min_size=min_size)
        fastremap.renumber(new_mask, in_place=True)
        return new_mask

# Not so fast after all use the quick-clean and clustering as above
def fast_reconstruction(net_output, cell_prob_threshold=0.3, clean_mask=True,
            flow_threshold=0.4, min_size=15, hole_size=3, device="cpu"):
    """
    Fast reconstruction from the omnipose outputs
    Arguments:
    ---------
    net_output: (4, H, W) np.ndarray consisting of the following arrays at indices 0:2, 2, 3 resp.
        dP: (2, H, W) np.ndarray , flow-y and flow-x each (H, W) in size
        dist: (H, W) distance image predicted by the network
        boundary: (H, W) distance 
    
    cell_prob_threshold: (float, default: 0.3) , threshold used on the distance field,
                         to get cell probability binary mask
    clean_mask: (bool, default: True) if True cleans the mask and removes bad mask with
                flows errors greater than flow_threshold (default: 0.4).
                clean_mask also removes smaller masks and fill holes
    
    flow_threshold: (float, default: 0.4), threshold used on the mean-squared error between
                calculated flows from masks generated by the network output reconstruction
                and flows predicted by the network.
    
    min_size: (int, default: 15), minimum size of the cell mask in pixels
    hole_size: (int, default: 3), holes of size in percent that are filled 
    Returns:
    --------
    masks: (H, W) labelled array of cell masks,
            0=background, 1,2,3 ... cell mask numbers
    """
    dP = net_output[:2]
    dist = net_output[2]
    boundary = net_output[3]

    # calculate the binary threshold
    cp_mask = dist > cell_prob_threshold
    # scale the divergence
    dP_scaled = divergence_rescale(dP, cp_mask)

    if device == "cpu":
    # interate and trace the flows back to origin points
        traced_p, inds = follow_flows_cpu_omni(dP_scaled, cp_mask, interp=False)
    elif device[4:] == "cuda":
        traced_p, inds = follow_flows_gpu_omni(dP_scaled, cp_mask, device=device)

    newinds = traced_p[:, inds[:, 0], inds[:, 1]].swapaxes(0, 1)

    new_mask = np.zeros((traced_p.shape[1], traced_p.shape[2]))

    
    newinds = np.rint(newinds).astype(int)
    skelmask = np.zeros_like(dist, dtype=bool)
    skelmask[newinds[:, 0], newinds[:, 1]] = 1

    # disconnect skeletons at the edge, 5, pixels in
    border_mask = np.zeros(skelmask.shape, dtype=bool)
    border_px = border_mask.copy()
    border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

    border_px[border_mask] = skelmask[border_mask]

    border_px[boundary > -1] = 0

    skelmask[border_mask] = border_px[border_mask]

    LL = label(skelmask, connectivity=1)

    new_mask[inds[:, 0], inds[: , 1]] = LL[newinds[:, 0], newinds[:, 1]]
    

    return new_mask

def format_labels(labels, clean=False):

    labels = labels.astype('int32')
    labels -= np.min(labels)
    labels = labels.astype('uint32')

    if clean:
        # They do an optional clean to remove small area susing regionprops
        pass

    fastremap.renumber(labels, in_place=True)
    labels = fastremap.refit(labels)
    return labels

def clean_boundary(labels,boundary_thickness=3,area_thresh=30):
    """
    Delete boundary masks below a given size threshold. Default boundary thickness is 3px,
    meaning masks that are 3 or fewer pixels from the boudnary will be candidates for removal. 
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=boundary_thickness)
    clean_labels = np.copy(labels)
    for cell_ID in np.unique(labels):
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= 0.5: #only premove cells that are 50% or more edge px
            clean_labels[mask] = 0
    return clean_labels



# function for plotting copied from cellpose/plots.py
def flows_to_colors(dP,transparency=False,mask=None):
    """ dP is 2 x Y x X => 'optic' flow representation 
    
    Parameters
    -------------
    
    dP: 2xLyxLx array
        Flow field components [dy,dx]
        
    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)
        
    mask: 2D array 
        Multiplies each RGB component to suppress noise
    
    """
    
    dP = np.array(dP)
    mag = np.clip(normalize99(np.sqrt(np.sum(dP**2,axis=0))), 0, 1.)
    angles = np.arctan2(dP[1], dP[0])+np.pi
    a = 2
    r = ((np.cos(angles)+1)/a)
    g = ((np.cos(angles+2*np.pi/3)+1)/a)
    b =((np.cos(angles+4*np.pi/3)+1)/a)
    
    if transparency:
        im = np.stack((r,g,b,mag),axis=-1)
    else:
        im = np.stack((r*mag,g*mag,b*mag),axis=-1)
        
    if mask is not None and transparency and dP.shape[0]<3:
        im[:,:,-1] *= mask
        
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im


def single_flow_to_color(flow, mask, type='Y'):

    mag = np.clip(normalize99(np.sqrt(np.sum(flow**2))), 0, 1.)
    if type == 'Y':
        angle = np.arcsin(flow) + np.pi
    else:
        angle = np.arccos(flow) + np.pi
    a = 2
    r = ((np.sin(angle)+1)/a)
    g = ((np.sin(angle+2*np.pi/3)+1)/a)
    b =((np.sin(angle+4*np.pi/3)+1)/a)
    
    
    im = np.stack((r*mag,g*mag,b*mag),axis=-1)
    
    mask = mask > 0
    mask_rgb = np.stack((mask, mask, mask), axis=-1)
    
    im[~mask_rgb] = 1
    
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def generate_weights(filename, sigma=5, w0=10):

    img = io.imread(filename)
    # removing objects and calculating distances to objects needs labelled images
    labeledImg, num = label(img, return_num=True, connectivity=2)
    # remove small objects
    labeledImg = remove_small_objects(labeledImg, min_size=250)
    # unique values == number of blobs
    unique_values = np.unique(labeledImg) 
    num_values = len(unique_values)
    h, w = labeledImg.shape
    # stack keeps distance maps each blob
    stack = np.zeros(shape=(num_values, h, w))
    for i in range(num_values):
        stack[i] = ndi.distance_transform_edt(~(labeledImg == unique_values[i]))
    # sort the distance
    sorted_distance_stack = np.sort(stack, axis=0)
    # d1 and d2 are the shortest and second shortest distances to each object, 
    # sorted_distance_stack[0] is distance to the background. One can ignore it
    distance_sum = sorted_distance_stack[1] + sorted_distance_stack[2]
    squared_distance = distance_sum ** 2/ (2 * (sigma**2))
    weightmap = w0 * np.exp(-squared_distance)*(labeledImg == 0)
    return weightmap


def to_cpu(tensor):
    return tensor.detach().cpu()

def plot_results_batch(phase_batch, predictions_batch):
    """
    Gives figures handles for plotting results to tensorboard

    Args:
        phase_batch: numpy.ndarray (B, C, H, W), C=1 in our case
        predictions_batch: numpy.ndarray(B, C, H, W), C = 1 or 2 for Unet, more for omni
        Returns:
        fig_handles: a list of B fig handles, where each figure has bboxes
                     plotted on them appropriately 
    """
    # return a list of matplotlib figure objects
    fig_handles = []
    B, n_outputs, _, _ = predictions_batch.shape

    for i in range(B):
        fig, ax = plt.subplots(nrows=1 + n_outputs, ncols=1)
        fig.tight_layout()
        ax[0].imshow(phase_batch[i][0], cmap='gray')
        for j in range(n_outputs):
            ax[1 + j].imshow(predictions_batch[i][j], cmap='gray')
        fig_handles.append(fig)
    return fig_handles