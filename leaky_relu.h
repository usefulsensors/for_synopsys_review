/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LEAKY_RELU_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LEAKY_RELU_H_

#include "tensorflow/lite/c/common.h"
#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#define TEST_ARC_MLI
namespace tflite {

// Input/output tensor index.
extern const int kInputTensor;
extern const int kOutputTensor;

struct LeakyReluOpData {
  // quantization parameters
  int32_t output_multiplier_alpha;
  int32_t output_shift_alpha;
  int32_t output_multiplier_identity;
  int32_t output_shift_identity;
  int32_t input_zero_point;
  int32_t output_zero_point;
#ifdef TEST_ARC_MLI
  // The result of checking if MLI optimized version of tensors can be used.
  bool is_mli_applicable;
  // Tensors in MLI format.
  mutable ops::micro::MliTensorInterface mli_in;
  mutable ops::micro::MliTensorInterface mli_out;
  mutable ops::micro::MliTensorInterface mli_slope_coeff;
#endif
};

TfLiteStatus CalculateOpDataLeakyRelu(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus LeakyReluPrepare(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LEAKY_RELU_H_
