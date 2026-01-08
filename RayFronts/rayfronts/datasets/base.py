"""Defines abstract base classes for all datasets/datasources."""

import abc
from typing import Tuple, Union

import torch


class PosedRgbdDataset(torch.utils.data.IterableDataset, abc.ABC):
  """A base interface for loading from any posed RGBD source.
  
  Attributes:
    intrinsics_3x3:  A 3x3 float tensor including camera intrinsics. This
      must be set by the child class.
    rgb_h: RGB image height to resize output to. If -1, no resizing is done.
    rgb_w: RGB image width to resize output to. If -1, no resizing is done.
    depth_h: Depth image height to resize output to. If -1, no resizing is done.
    depth_w: Depth image width to resize output to. If -1, no resizing is done.
    frame_skip: See __init__.
    interp_mode: See __init__.
  """

  def __init__(self,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    """
    Args:
      rgb_resolution: Resolution of rgb frames. Set to None to keep the same
        size as original. Either a single integer or a tuple (height, width).
      depth_resolution: Resolution of depth frames. Set to None to keep the same
        size as original. Either a single integer or a tuple (height, width).
      frame_skip: Frame skipping when loading data. Ex. frame_skip: 2, means we
        consume a frame then drop 2 and so on.
      interp_mode: Which pytorch interpolation mode for rgb and feature
        interpolation (Depth and Segmentation always use nearest-exact).
    """

    self.intrinsics_3x3 = None
    if isinstance(rgb_resolution, int):
      self.rgb_h = rgb_resolution
      self.rgb_w = rgb_resolution
    elif hasattr(rgb_resolution, "__len__"):
      self.rgb_h, self.rgb_w = rgb_resolution
    else:
      self.rgb_h = -1
      self.rgb_w = -1

    if isinstance(depth_resolution, int):
      self.depth_h = depth_resolution
      self.depth_w = depth_resolution
    elif hasattr(depth_resolution, "__len__"):
      self.depth_h, self.depth_w = depth_resolution
    else:
      self.depth_h = -1
      self.depth_w = -1

    self.frame_skip = frame_skip
    self.interp_mode = interp_mode

  @abc.abstractmethod
  def __iter__(self):
    """Iterater returning posed RGBD frames in order

    Returns:
      A dict mapping keys {rgb_img, depth_img, pose_4x4} to tensors of shapes
      {3xHxW, 1xH'xW', 4x4} respectively. RGB images are in float32 and have
      (0-1) range. Depth images contain positive values with possible NaNs for
      non valid depth values, +Inf for too far, -Inf for too close. 
      Pose is a 4x4 float32 tensor in opencv RDF. a pose is the extrinsics
      transformation matrix that takes you from camera/robot coordinates to
      world coordinates. Last row should always be [0, 0, 0, 1].

      A confidence map with key {confidence_map} may or may not be included.
      The map is 1xH'xW' float32 tensors in range [0-1] where 0 is least
      confident and 1 is most confident.

      A time stamp key {ts} may or may not be returned with its corresponding
      float32 tensor of shape 1 containing seconds since the epoch.
    """
    pass

class SemSegDataset(PosedRgbdDataset, abc.ABC):
  """Base interface for datasets that provide semantic label images as well."""

  @property
  @abc.abstractmethod
  def num_classes(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def cat_id_to_name(self):
    """Returns a mapping from class id to class name.
    
    cat_id_to_name[cat_id] gives the name of that class id.
    """
    pass

  @abc.abstractmethod
  def __iter__(self):
    """Iterater returning posed RGBD frames + semantic segmentaion in order.

    Returns:
      Returns the same items as PosedRgbdDataset.
      In addition semantic segmentation is returned with key {semseg_img} as
      a 1xHxW long tensor specifying class ids for each pixel plus the `0` value
      for no semantic label. The user of the class should be able to get each
      class id correspondance to the class name through the cat_id_to_name 
      property
    """
    pass
