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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/leaky_relu.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/leaky_relu.h"

namespace tflite {

// Input/output tensor index.
const int kInputTensor = 0;
const int kOutputTensor = 0;
#ifdef TEST_ARC_MLI
bool IsMliApplicable(TfLiteContext* context, const TfLiteTensor* input) {
  // MLI optimized version only supports int8_t datatype
  return (input->type == kTfLiteInt8);
}
#endif
TfLiteStatus CalculateOpDataLeakyRelu(TfLiteContext* context,
                                      TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));


  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  LeakyReluOpData* data = static_cast<LeakyReluOpData*>(node->user_data);
  const auto* params =
      static_cast<TfLiteLeakyReluParams*>(node->builtin_data);
      float alpha_multiplier;
  if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {



    data->input_zero_point = input->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    int output_shift_alpha;
    alpha_multiplier = static_cast<double>(
        input->params.scale * params->alpha / output->params.scale);
    QuantizeMultiplier(alpha_multiplier, &data->output_multiplier_alpha,
                       &output_shift_alpha);
    data->output_shift_alpha = static_cast<int32_t>(output_shift_alpha);

    int output_shift_identity;
    double identity_multiplier =
        static_cast<double>(input->params.scale / output->params.scale);
    QuantizeMultiplier(identity_multiplier, &data->output_multiplier_identity,
                       &output_shift_identity);
    data->output_shift_identity = static_cast<int32_t>(output_shift_identity);
  //  MicroPrintf("%d\n",data->input_zero_point);
//    MicroPrintf("%d\n",data->output_zero_point);
  //  MicroPrintf("input->params.scale%d\n",input->params.scale);
  //  MicroPrintf("output->params.scale %d\n",output->params.scale);
  }
#ifdef TEST_ARC_MLI

  data->is_mli_applicable = IsMliApplicable(context, input);
  if (data->is_mli_applicable) {
    MicroPrintf("\nEntering to prepare of is_mli_applicable");
    data->mli_in = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
    data->mli_out = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
    data->mli_slope_coeff = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
//{Q3.4
MicroPrintf("params->alpha %f\n",params->alpha);

int8_t val_i8 = (int8_t)(params->alpha*(1<<7));
//int8_t val_i8 = (int8_t)(params->alpha*(255));
MicroPrintf("fixed 8: %x\n", val_i8);
#if 0
TfLiteTensor tensor;
tensor.data.data = (int8_t *)&val_i8;
tensor.dims = {-1};
tensor.bytes = 1;
tensor.type = kTfLiteInt8;
tensor.params.scale = 0;
tensor.params.zero_point = 0;
MicroPrintf("tensor->data.int8:%x",*(tensor.data.int8));
MicroPrintf("tensor->data.int8:%x",(tensor.data.int8));
#endif
//}
    TfLiteTensor* slope_coeffs_tensor =
        context->GetTensor(context, params->alpha);
        slope_coeffs_tensor->data.data = (int8_t *)&val_i8;
      //  MicroPrintf("fixed 8: %x\n", slope_coeffs_tensor->data.data);
      //  MicroPrintf("fixed 8: %x\n", (int8_t *)&(slope_coeffs_tensor->data.data));
        //slope_coeffs_tensor->is_variable = true;
      //  slope_coeffs_tensor->params.scale = input->params.scale;
      //  slope_coeffs_tensor->params.zero_point = input->params.zero_point;
    ops::micro::ConvertToMliTensor(input, &data->mli_in);
    ops::micro::ConvertToMliTensor(output, &data->mli_out);
    ops::micro::ConvertToMliTensor(slope_coeffs_tensor, &data->mli_slope_coeff);
    /* Flatten tensors to simplify the process (as we don't support
     * broadcasting). */
    data->mli_in.Shape()[0] =
        mli_hlp_count_elem_num(data->mli_in.MliTensor(), 0);
    data->mli_slope_coeff.Shape()[0] =
        mli_hlp_count_elem_num(data->mli_slope_coeff.MliTensor(), 0);
    data->mli_out.Shape()[0] =
        mli_hlp_count_elem_num(data->mli_out.MliTensor(), 0);
    data->mli_in.MemStride()[0] = data->mli_slope_coeff.MemStride()[0] = 1;
    data->mli_out.MemStride()[0] = 1;
    *data->mli_in.Rank() = *data->mli_slope_coeff.Rank() = 1;
    *data->mli_out.Rank() = 1;

    //ops::micro::ConvertToMliTensor(&tensor, &data->mli_slope_coeff);
    MicroPrintf("tensor->data.int8:%x",*(slope_coeffs_tensor->data.int8));
    MicroPrintf("\n Exiting to prepare of is_mli_applicable");

  }
#endif

  return kTfLiteOk;
}

TfLiteStatus LeakyReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpDataLeakyRelu(context, node);
}

}  // namespace tflite
