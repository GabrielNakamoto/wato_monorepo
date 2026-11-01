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

#include "bevfusion/bevfusion_core.hpp"

#include <dlfcn.h>
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace wato::perception::bevfusion
{

// TRT logging
namespace
{
static_assert(NV_TENSORRT_MAJOR >= 10, "This build path targets TensorRT 10/11 only.");

class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }
};

TrtLogger g_trt_logger;
}  // namespace

BEVFusionCore::BEVFusionCore(::bevfusion::CoreParameter param, BEVFusionONNXConfig onnx_config)
: param_(param), onnx_config_(onnx_config)
{}

BEVFusionCore::~BEVFusionCore()
{
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

bool BEVFusionCore::isInitialized() const
{
  return initialized_;
}

bool BEVFusionCore::hasCalibration() const
{
  return has_calibration_;
}

const SegMap & BEVFusionCore::getSegMap() const
{
  return seg_map_;
}

bool BEVFusionCore::initialize()
{
  // Must load this plugin before create_core() — TRT will fail to deserialize head.bbox.plan without it
  if (dlopen("libcustom_layernorm.so", RTLD_NOW) == nullptr) {
    std::cerr << "[BEVFusionCore] ERROR: Failed to load libcustom_layernorm.so: " << dlerror() << std::endl;
    return false;
  }

  // Check if all required model files exist, and build missing TensorRT engines if needed
  if (!checkModelFilesExist()) {
    std::cout << "[BEVFusionCore] Building TRT engines. Please hold for several minutes..." << std::endl;
    if (!buildTRTEngines()) {
      std::cerr << "[BEVFusionCore] ERROR: Failed to build TensorRT engines" << std::endl;
      return false;
    }
  }

  // Ensure that all required model files exist after the build step
  if (!checkModelFilesExist()) {
    std::cerr << "[BEVFusionCore] ERROR: Missing plan/onnx files, cannot create core!" << std::endl;
    return false;
  }

  std::cout << "[BEVFusionCore] All required model files exist. Creating core..." << std::endl;

  pipeline_ = ::bevfusion::create_core(param_);
  if (pipeline_ == nullptr) {
    std::cerr << "[BEVFusionCore] ERROR: Failed to create BEVFusion pipeline. Ensure model files exist and GPU memory "
                 "is sufficient."
              << std::endl;
    return false;
  }

  // dedicated stream: BEVFusion's GPU ops stay ordered in their own queue, not on the null (default) stream shared by everything else
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
    std::cerr << "[BEVFusionCore] ERROR: cudaStreamCreate failed." << std::endl;
    pipeline_.reset();
    return false;
  }

  // Expected nuScenes map grid: x, y in [-50, 50) m at 0.5 m per pixel
  seg_map_.resolution = 0.5f;
  seg_map_.origin_x = -50.0f;
  seg_map_.origin_y = -50.0f;

  initialized_ = true;
  return true;
}

std::vector<BoundingBox> BEVFusionCore::infer(
  const std::vector<const unsigned char *> & camera_images, const std::vector<float> & lidar_points, int num_points)
{
  if (!initialized_) {
    std::cerr << "[BEVFusionCore] ERROR: infer called before initialization." << std::endl;
    return {};  // guard check, return empty vector if pipeline is not initialized
  }
  if (!has_calibration_) {
    std::cerr << "[BEVFusionCore] ERROR: infer called before calibration was updated." << std::endl;
    return {};
  }

  // Ensure images are not empty
  for (size_t i = 0; i < camera_images.size(); ++i) {
    if (camera_images[i] == nullptr) {
      std::cerr << "[BEVFusionCore] ERROR: camera_images[" << i << "] is null" << std::endl;
      return {};
    }
  }

  // Ensure lidar points are of expected size.
  const size_t expected_lidar_floats = static_cast<size_t>(num_points) * static_cast<size_t>(param_.lidar_scn.voxelization.num_feature);
  if (lidar_points.size() != expected_lidar_floats) {
    std::cerr << "[BEVFusionCore] ERROR: lidar_points.size() (" << lidar_points.size()
              << ") != num_points * num_features (" << expected_lidar_floats << ")" << std::endl;
    return {};
  }

  // Convert LiDAR points to FP16.
  // Notes:
  // - lidar_points.size() is the total # of floats in the flat array of 5 features per lidar point
  // - lidar_half_buf_ is a vector of __half and not nvtype::half because __float2half returns __half format
  // - lidar_half_buf_ is a pre-allocated member: resize() is a no-op when size hasn't grown,
  //   avoiding per-frame heap allocation/deallocation for this potentially-large buffer
  lidar_half_buf_.resize(lidar_points.size());
  for (size_t i = 0; i < lidar_points.size(); ++i) {
    lidar_half_buf_[i] = __float2half(lidar_points[i]);
  }

  // Bevfusion forward pass call
  // Notes:
  // - camera_images.data(): gives the array of image pointers. Each pointer points to the start of the data for one camera.
  //   The `const_cast` is used because the vendor library expects a non-const pointer, even though it doesn't modify the image data.
  // - lidar_half_buf_.data(): gives a pointer to the first element of the pre-allocated FP16 vector.
  //   The `reinterpret_cast` is used to cast this pointer to `const nvtype::half*`, which is the expected type for the vendor library API.
  //   Could have also used memcpy to manually copy bits from __half to nvtype::half since they are bitwise identical.
  // - Use .data() for underlying array
  // - returns pair of (1) 3d bounding box detections (2) (optional) classified 2d map segmentations
  auto head_outputs = pipeline_->forward(
    const_cast<const unsigned char **>(camera_images.data()),
    reinterpret_cast<const nvtype::half *>(lidar_half_buf_.data()),
    num_points,
    stream_);

  auto detections = head_outputs.first;

  // map vendor type to our bounding box type
  std::vector<BoundingBox> result;
  // NOTE: SwinT doesn't have detection head, only segmentation outputs for dev testing currently
  if (! param_.segmap.enabled) {
    result.reserve(detections.size());
    for (const auto & d : detections) {
      BoundingBox box;
      box.position = {d.position.x, d.position.y, d.position.z};
      box.size = {d.size.w, d.size.l, d.size.h};
      box.velocity = {d.velocity.vx, d.velocity.vy};
      box.z_rotation = d.z_rotation;
      box.score = d.score;
      box.id = d.id;
      result.push_back(box);
    }
  }

  // update segmentation map
  if (param_.segmap.enabled) {
    auto map = head_outputs.second;
    seg_map_.num_classes = map.classes;
    seg_map_.height = map.height;
    seg_map_.width = map.width;
    seg_map_.masks.assign(map.data, map.data + map.numel());
  }
  return result;
}

void BEVFusionCore::updateCalibration(
  const std::vector<float> & camera_to_lidar,
  const std::vector<float> & camera_intrinsics,
  const std::vector<float> & lidar_to_image_projection,
  const std::vector<float> & img_aug_matrix)
{
  if (!initialized_) {
    std::cerr << "[BEVFusionCore] ERROR: updateCalibration called before initialization." << std::endl;
    return;
  }
  pipeline_->update(
    camera_to_lidar.data(), camera_intrinsics.data(), lidar_to_image_projection.data(), img_aug_matrix.data(), stream_);
  has_calibration_ = true;
}

bool BEVFusionCore::checkModelFilesExist() const
{
  bool result = true;
  if (!std::filesystem::exists(param_.camera_model)) {
    std::cerr << "[BEVFusionCore] MISSING: camera_backbone_plan='" << param_.camera_model << "'" << std::endl;
    result = false;
  }
  if (!std::filesystem::exists(param_.camera_vtransform)) {
    std::cerr << "[BEVFusionCore] MISSING: camera_vtransform_plan='" << param_.camera_vtransform << "'" << std::endl;
    result = false;
  }
  if (!std::filesystem::exists(param_.transfusion)) {
    std::cerr << "[BEVFusionCore] MISSING: fuser_plan='" << param_.transfusion << "'" << std::endl;
    result = false;
  }
  if (param_.segmap.enabled) {
    if (!std::filesystem::exists(param_.segmap.model)) {
      std::cerr << "[BEVFusionCore] MISSING: head_seg_plan='" << param_.segmap.model << "'" << std::endl;
      result = false;
    }
  } else if (!std::filesystem::exists(param_.transbbox.model)) {
    std::cerr << "[BEVFusionCore] MISSING: head_bbox_plan='" << param_.transbbox.model << "'" << std::endl;
    result = false;
  }
  if (!std::filesystem::exists(onnx_config_.lidar_backbone)) {
    std::cerr << "[BEVFusionCore] MISSING: lidar_backbone_onnx='" << onnx_config_.lidar_backbone << "'" << std::endl;
    result = false;
  }
  return result;
}

bool BEVFusionCore::buildTRTEngines() const
{
  const std::filesystem::path build_dir(onnx_config_.build_dir);

  std::cout << "[BEVFusionCore][buildTRTEngines] Detected TensorRT version: " << NV_TENSORRT_MAJOR << "."
            << NV_TENSORRT_MINOR << std::endl;

  if (!compileTrtModel("camera.backbone", onnx_config_.camera_backbone, build_dir / "camera.backbone.plan")) {
    return false;
  }
  if (!compileTrtModel("fuser", onnx_config_.fuser, build_dir / "fuser.plan")) {
    return false;
  }
  if (!compileTrtModel("camera.vtransform", onnx_config_.camera_vtransform, build_dir / "camera.vtransform.plan")) {
    return false;
  }
  if (param_.segmap.enabled) {
    if (!compileTrtModel("head.seg", onnx_config_.head_segmap, build_dir / "head.seg.plan")) {
      return false;
    }
  } else if (!compileTrtModel("head.bbox", onnx_config_.head_bbox, build_dir / "head.bbox.plan")) {
    return false;
  }

  return true;
}

bool BEVFusionCore::compileTrtModel(
  const std::string & name, const std::filesystem::path & onnx_file, const std::filesystem::path & plan_file) const
{
  // Pre-checks
  if (std::filesystem::exists(plan_file)) {
    std::cout << "[BEVFusionCore][compileTrtModel] Model " << plan_file << " already built!" << std::endl;
    return true;
  }
  if (!std::filesystem::exists(onnx_file)) {
    std::cerr << "[BEVFusionCore][compileTrtModel] ERROR: The model [" << onnx_file << "] does not exist." << std::endl;
    return false;
  }

  // Create build directory
  std::error_code ec;
  std::filesystem::create_directories(plan_file.parent_path(), ec);
  if (ec) {
    std::cerr << "[BEVFusionCore] ERROR: Failed to create build directory: " << ec.message() << std::endl;
    return false;
  }

  // Create TRT Builder
  std::cout << "[BEVFusionCore][compileTrtModel] Building the model: " << plan_file << std::endl;
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_trt_logger));
  if (!builder) {
    std::cerr << "[BEVFusionCore][compileTrtModel] ERROR: Failed to create TRT builder for " << name << std::endl;
    return false;
  }

  // Create empty network to load the model into
  // TRT10+: explicit batch is the only mode, no creation flag needed/available.
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));

  // Parse the ONNX model file
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, g_trt_logger));
  if (!parser->parseFromFile(onnx_file.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "[BEVFusionCore][compileTrtModel] ERROR: Failed to parse ONNX: " << onnx_file << std::endl;
    return false;
  }

  // Set builder configurations
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2048ULL * 1024 * 1024);
  config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

  // These ONNX graphs are exported with explicit Q/DQ nodes (PTQ). On TRT10, the builder needs
  // kFP16/kINT8 explicitly or it runs those Q/DQ pairs as literal FP32 quantize/dequantize math
  // instead of fusing them into INT8 kernels. TRT11+ removed those flags — networks are always
  // strongly typed there, so precision comes straight from the ONNX graph's own tensor types.
#if NV_TENSORRT_MAJOR < 11
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  if (param_.lidar_scn.precision == ::bevfusion::lidar::Precision::Int8) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
#endif

  // Force allowed tensor formats on inputs and outputs
  for (int i = 0; i < network->getNbInputs(); ++i) {
    network->getInput(i)->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
  }
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    network->getOutput(i)->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
  }

  // Build serialized engine plan
  std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
  if (!serialized) {
    std::cerr << "[BEVFusionCore][compileTrtModel] ERROR: Failed to build model " << plan_file << std::endl;
    return false;
  }

  // Save the serialized engine to the plan file
  std::ofstream out(plan_file, std::ios::binary);
  if (!out) {
    std::cerr << "[BEVFusionCore][compileTrtModel] ERROR: Failed to open " << plan_file << " for writing." << std::endl;
    return false;
  }
  out.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
  return out.good();
}

}  // namespace wato::perception::bevfusion
