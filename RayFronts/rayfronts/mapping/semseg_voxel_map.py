"""Module defining a mapper that lifts 2D semantic segmentation labels to 3D.

Typical usage example:

  map = SemSegVoxelMap(intrinsics_3x3, None, visualizer, encoder)
  for batch in dataloader:
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
    semseg_img = batch["semseg_img"].cuda()
    map.process_posed_rgbd(rgb_img, depth_img, pose_4x4, semseg_img=semseg_img)
  map.vis_map()

  map.save("test.pt")
"""

from typing_extensions import override, List, Tuple

import torch

from rayfronts.mapping.base import RGBDMapping
from rayfronts import geometry3d, image_encoders, visualizers
from rayfronts.utils import compute_cos_sim

class SemSegVoxelMap(RGBDMapping):
  """A special mapper to lift 2D semantic segmentation labels to 3D voxels.
  
  The mapper stores counts at each voxel for each class. The class with the 
  highest count is determined to be the class for that voxel. This also includes
  the ignore class which is always set to have index 0.

  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.
    clip_bbox: See base.

    interp_mode: See __init__.
    max_pts_per_frame: See __init__.
    vox_size: See __init__.

    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_onehot: (Nx(3+C)) Float tensor; 3 for rgb, C for number of
      classes, This tensor is aligned with global_vox_xyz.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               interp_mode: str = "bilinear",
               max_pts_per_frame: int = -1,
               vox_size: int = 1,
               vox_accum_period: int = 1):
    """
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      interp_mode: Interpolation used for projecting RGB.
      max_pts_per_frame: How many points to project per frame. Set to -1 to 
        project all valid depth points.
      vox_size: Length of a side of a voxel in meters.
      vox_accum_period: How often do we aggregate voxels into the map.
    """

    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox)

    self.visualizer = visualizer

    self.max_pts_per_frame = max_pts_per_frame

    self.vox_size = vox_size
    self.interp_mode = interp_mode

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+C)) 3 for rgb, C for number of classes
    self.global_vox_rgb_onehot = None

    # Temporary point cloud accumulation before voxelization
    self.vox_accum_period = vox_accum_period
    self.vox_accum_cnt = 0
    self.tmp_pc_xyz = list()
    self.tmp_pc_rgb_onehot = list()

  @property
  def global_vox_rgb(self):
    if self.global_vox_rgb_onehot is not None:
      return self.global_vox_rgb_onehot[:, :3]
    else:
      return None

  @property
  def global_vox_onehot(self):
    if self.global_vox_rgb_onehot is not None:
      return self.global_vox_rgb_onehot[:, 3:]
    else:
      return None

  @property
  def global_vox_preds(self):
    if self.global_vox_rgb_onehot is not None:
      return torch.argmax(self.global_vox_onehot, dim=-1)
    else:
      return None

  @override
  def save(self, file_path):
    torch.save(dict(global_vox_xyz=self.global_vox_xyz,
                    global_vox_rgb_onehot=self.global_vox_rgb_onehot),
                    file_path)

  @override
  def load(self, file_path):
    d = torch.load(file_path, weights_only=True)
    self.global_vox_xyz = d["global_vox_xyz"].to(self.device)
    self.global_vox_rgb_onehot = \
      d["global_vox_rgb_onehot"].to(self.device)

  @override
  def is_empty(self) -> bool:
    return self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         semseg_img: torch.LongTensor = None) -> dict:
    """Consumes a posed rgbd+semantic segmentation image to update the map. 

    Args:
      rgb_img: A Bx3xHxW RGB float tensor within the (0-1) range describing a 
        batch of images
      depth_img: A Bx1xH'xW' float tensor with values > 0 describing a batch of 
        depth images. May include NaN and Inf values which will be ignored.
      pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
        a pose is the extrinsics transformation matrix that takes you from
        camera/robot coordinates to world coordinates.
      conf_map: (Optional) A Bx1xH'xW' float tensor with values in [0-1]
        with 1 being most confident in the depth value.
      semseg_img: One hot encoded semantic masks as BxCxHxW tensor
        where C is number of classes.
    """
    update_info = dict()

    assert semseg_img is not None

    pts_xyz, selected_indices = geometry3d.depth_to_pointcloud(
      depth_img, pose_4x4, self.intrinsics_3x3,
      conf_map=conf_map,
      max_num_pts=self.max_pts_per_frame
    )

    B, _, rH, rW = rgb_img.shape
    B, _, dH, dW = depth_img.shape

    if rH != dH or rW != dW:
      pts_rgb = torch.nn.functional.interpolate(
        rgb_img,
        size=(dH, dW),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"])
      semseg_img = torch.nn.functional.interpolate(
        semseg_img.float(),
        size=(dH, dW),
        mode="nearest-exact").long()
    else:
      pts_rgb = rgb_img

    pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]

    NC = semseg_img.shape[1]
    pts_onehot = \
      semseg_img.permute(0, 2, 3, 1).reshape(-1, NC)[selected_indices]

    N = pts_rgb.shape[0]
    pts_rgb_onehot = torch.cat((pts_rgb, pts_onehot), dim=-1)

    self.tmp_pc_rgb_onehot.append(pts_rgb_onehot)
    self.tmp_pc_xyz.append(pts_xyz)
    self.vox_accum_cnt += B

    if self.vox_accum_cnt >= self.vox_accum_period:
      self.vox_accum_cnt = 0

      if self.global_vox_xyz is not None:
        self.tmp_pc_xyz.append(self.global_vox_xyz)
        self.tmp_pc_rgb_onehot.append(self.global_vox_rgb_onehot)

      pts_xyz = torch.cat(self.tmp_pc_xyz)
      pts_rgb_onehot = torch.cat(self.tmp_pc_rgb_onehot, dim=0)
      self.tmp_pc_xyz.clear()
      self.tmp_pc_rgb_onehot.clear()

      vox_xyz, vox_rgb_onehot, vox_cnt = \
        geometry3d.pointcloud_to_sparse_voxels(
          pts_xyz, feat_pc=pts_rgb_onehot,
          vox_size=self.vox_size, return_counts=True,
          aggregation="sum")

      vox_rgb_onehot[:, :3] = vox_rgb_onehot[:, :3] / vox_cnt
      self.global_vox_xyz = vox_xyz
      self.global_vox_rgb_onehot = vox_rgb_onehot

    return update_info

  @override
  def vis_map(self):
    if self.visualizer is None or self.is_empty():
      return
    self.visualizer.log_pc(self.global_vox_xyz, self.global_vox_rgb,
                           layer="voxel_rgb")
    self.visualizer.log_label_pc(self.global_vox_xyz, self.global_vox_preds,
                                 layer="voxel_labels")

  @override
  def vis_update(self, **kwargs) -> None:
    pass
