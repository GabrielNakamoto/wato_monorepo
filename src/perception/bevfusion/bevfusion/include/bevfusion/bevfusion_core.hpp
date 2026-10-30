// Copyright (c) 2025-present WATonomous. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <bevfusion/bevfusion.hpp>
#include <opencv2/core/mat.hpp>

namespace wato::perception::bevfusion
{
/** @brief ONNX weight paths and segmentation layout for a specific model version. */
struct BEVFusionONNXConfig
{
  std::string build_dir;
  std::string lidar_backbone;
  std::string head_bbox;
  std::string head_segmap;
  std::string camera_backbone;
  std::string camera_vtransform;
  std::string fuser;
  bool seg_enabled = false;
  int seg_num_classes = 6;
};

/**
 * @brief Top-down segmentation map around the LiDAR, one binary mask per class.
 *
 * Grid convention (LiDAR frame, x forward, y left):
 *   - row indexes x, col indexes y
 *   - pixel (row, col) covers x in [origin_x + row * resolution, origin_x + (row + 1) * resolution),
 *     and likewise for y with col
 *   - masks[c * height * width + row * width + col] is 1 if class c is present, else 0
 */
struct SegMap
{
  int num_classes = 0;
  int height = 0;  // rows (x direction)
  int width = 0;  // cols (y direction)
  float resolution = 0.0f;  // meters per pixel
  float origin_x = 0.0f;  // LiDAR-frame x of the min corner of pixel (0, 0), meters
  float origin_y = 0.0f;  // LiDAR-frame y of the min corner of pixel (0, 0), meters
  std::vector<uint8_t> masks;
};

/**
 * @brief Build a placeholder SegMap with a known asymmetric pattern, used until the CUDA seg head is available.
 *
 * Uses the expected nuScenes map grid (200x200, 0.5 m, centered on the LiDAR). Class 0 (drivable_area) is a
 * corridor only in front of the vehicle, and the last class (divider) is a line only on its left side, so a
 * flipped or rotated map is obvious in Foxglove.
 * @param num_classes Number of class masks to allocate
 * @return Placeholder segmentation map
 */
SegMap makePlaceholderSegMap(int num_classes);

/**
 * @brief 3D position of the bounding box
 */
struct Position
{
  float x, y, z;
};

/**
 * @brief Dimensions of the bounding box
 */
struct Size
{
  float w, l, h;  // x, y, z
};

/**
 * @brief Velocity of the bounding box
 */
struct Velocity
{
  float vx, vy;
};

/**
 * @brief 3D bounding box
 */
struct BoundingBox
{
  Position position;
  Size size;
  Velocity velocity;
  float z_rotation;
  float score;
  int id;
};

/**
 * @brief Core logic for BEVFusion.
 *
 * Note: Use ::bevfusion when referencing anything from the original CUDA-BEVFusion repository.
 * Use wato::perception::bevfusion (handled by namespaces) when referencing anything from wato's bevfusion node.
 *
 * Note: nvtype is defined in the original CUDA-BEVFusion repository under /CUDA-BEVFusion/src/common/dtype.hpp and
 * can be used as nvtype:: or ::nvtype::.
 */
class BEVFusionCore
{
public:
  /**
   * @brief Construct the core
   * @param param inference engine parameters of BEVFusion model
   * @param onnx_config file paths to ONNX weights for specific model layers
   */
  explicit BEVFusionCore(::bevfusion::CoreParameter param, BEVFusionONNXConfig onnx_config);
  ~BEVFusionCore();

  /**
   * @brief Create the CUDA-BEVFusion pipeline and deserialize TensorRT engines.
   * @return true if initialization was successful, false otherwise
   */
  bool initialize();

  /**
   * @brief Main inference entry point. Processes one synchronized frame of camera and LiDAR data.
   *        Converts data to required format and feed it to GPU pipeline (needs LiDAR data to be FP16).
   * @param camera_images Vector of camera images
   * @param lidar_points Vector of LiDAR points
   * @param num_points Number of LiDAR points
   * @return Vector of detected 3D bounding boxes
   *
   * @note lidar_points is a flattened 1D vector containing all the raw LiDAR data for a single frame
   *       - lidar_points.size() equals num_points * param_.lidar_scn.voxelization.num_feature.
   *       - The vector looks like [x1, y1, z1, i1, t1, x2, y2, z2, i2, t2, ...],
   *         where (x,y,z) are coordinates, i is intensity and t is the sweep time lag (0 for a single sweep).
   * @note Preconditions:
   *       - initialize() called and returned true; pipeline_ and stream_ are live.
   *       - updateCalibration() called at least once; pipeline needs camera extrinsics/intrinsics before its first forward pass
   *         to relate cameras to the LiDAR.
   */
  std::vector<BoundingBox> infer(
    const std::vector<const unsigned char *> & camera_images, const std::vector<float> & lidar_points, int num_points);

  /**
   * @brief Update the tansformation matrices mapping cameras to LiDARs
   * @param camera_to_lidar 4x4 camera to LiDAR transformation matrix
   * @param camera_instrinsics 3x3 camera intrinsics matrix
   * @param lidar_to_image_projection 4x4 LiDAR to image projection matrix
   * @param img_aug_matrix 4x4 image augmentation matrix
   *
   * @note Preconditions:
   *       - initialize() called and returned true; pipeline_ and stream_ are live.
   */
  void updateCalibration(
    const std::vector<float> & camera_to_lidar,
    const std::vector<float> & camera_intrinsics,
    const std::vector<float> & lidar_to_image_projection,
    const std::vector<float> & img_aug_matrix);

  /**
   * @brief Segmentation map from the most recent infer() call.
   * @return The latest SegMap; empty (num_classes == 0) if seg is disabled or infer() has not produced one
   * @note Currently returns makePlaceholderSegMap() output until the CUDA seg head lands in wato-cuda-bevfusion.
   */
  const SegMap & getSegMap() const;

  /**
   * @brief Check if the core is initialized
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const;

  /**
   * @brief Check if the calibration is initialized
   * @return true if calibration is initialized, false otherwise
   */
  bool hasCalibration() const;

private:
  /**
   * @brief Check if all model files exist
   * @return true if all model files exist, false otherwise
   */
  bool checkModelFilesExist() const;

  /**
   * @brief Build TRT engines
   * @return true if all engines were built successfully, false otherwise
   */
  bool buildTRTEngines() const;

  /**
   * @brief Compile TRT engine from ONNX file
   * @param name Name of the model
   * @param onnx_file Path to ONNX file
   * @param plan_file Path to plan file
   * @return true if successful, false otherwise
   */
  bool compileTrtModel(
    const std::string & name, const std::filesystem::path & onnx_file, const std::filesystem::path & plan_file) const;

  bool initialized_ = false;
  bool has_calibration_ = false;
  cudaStream_t stream_ = nullptr;
  std::shared_ptr<::bevfusion::Core>
    pipeline_;  // Reference to CUDA-BEVFusion pipeline, not wato's BEVFusionCore class.

  // Pre-allocated FP16 buffer for lidar point conversion (CPU side).
  // Reused across inference calls to avoid per-frame heap allocation.
  std::vector<__half> lidar_half_buf_;

  const ::bevfusion::CoreParameter param_;
  const BEVFusionONNXConfig onnx_config_;

  // Latest segmentation map, filled by infer() when the seg head is enabled.
  SegMap seg_map_;
};

}  // namespace wato::perception::bevfusion
