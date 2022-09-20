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

#include "tensorflow/lite/kernels/internal/reference/leaky_relu.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/leaky_relu.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {

#ifdef TEST_ARC_MLI
// Prepare MLI tensors and run Average or Max Pooling
TfLiteStatus EvalMli(TfLiteContext* context, const TfLiteLeakyReluParams* params,
                     const LeakyReluOpData& data, const TfLiteEvalTensor* input,
                     TfLiteEvalTensor* slope_coeffs,
                     TfLiteEvalTensor* output) {
  ops::micro::MliTensorAttachBuffer<int8_t>(input, &data.mli_in);
  ops::micro::MliTensorAttachBuffer<int8_t>(output, &data.mli_out);
  ops::micro::MliTensorAttachBuffer<int8_t>(slope_coeffs, &data.mli_slope_coeff);

  MicroPrintf("params->alpha:%f",params->alpha);

  // Tensors for data in fast (local) memory and config to copy data from
  // external to local memory
  mli_mov_cfg_t copy_config;
  mli_mov_cfg_for_copy(&copy_config);
  mli_tensor in_local = *data.mli_in.MliTensor();
  mli_tensor out_local = *data.mli_out.MliTensor();
  mli_tensor slope_coeffs_local = *data.mli_slope_coeff.MliTensor();

  ops::micro::MliTensorInterface in_local_interface(&in_local);
  ops::micro::MliTensorInterface out_local_interface(&out_local);
  ops::micro::MliTensorInterface slope_coeffs_local_interface(&slope_coeffs_local);


  TF_LITE_ENSURE_STATUS(get_arc_scratch_buffer_for_eltwise_tensors(
      context, &in_local_interface, &slope_coeffs_local_interface, &out_local_interface));

/* allocate the local buffers, and compute the slice size */
//TF_LITE_ENSURE(context, *in_local_interface.Rank() == 1 &&
//                            *slope_coeffs_local_interface.Rank() == 1 &&
  //                          *out_local_interface.Rank() == 1);

uint32_t min_capacity = *in_local_interface.DataCapacity();
min_capacity = std::min(min_capacity, *slope_coeffs_local_interface.DataCapacity());
min_capacity = std::min(min_capacity, *out_local_interface.DataCapacity());

const int slice_dim = 0;
const int slice_size =
    min_capacity / mli_hlp_tensor_element_size(out_local_interface.MliTensor());

/* is_local indicates that the tensor is already in local memory,
   so in that case the original tensor can be used,
   and there is no need to copy it to the local tensor*/
const bool input_is_local =
    in_local_interface.Data<int8_t>() == data.mli_in.Data<int8_t>();

const bool slope_coeffs_is_local =
    slope_coeffs_local_interface.Data<int8_t>() == data.mli_slope_coeff.Data<int8_t>();
const bool out_is_local =
    out_local_interface.Data<int8_t>() == data.mli_out.Data<int8_t>();

ops::micro::TensorSlicer input_slice(data.mli_in.MliTensor(), slice_dim,
                                      slice_size);
ops::micro::TensorSlicer slope_coeff_slice(data.mli_slope_coeff.MliTensor(), slice_dim,
                                      slice_size);
ops::micro::TensorSlicer out_slice(data.mli_out.MliTensor(), slice_dim,
                                   slice_size);

mli_tensor* input_tsr = input_is_local ? input_slice.Sub() : &in_local;
mli_tensor* slope_coeffs_tsr = slope_coeffs_is_local ? slope_coeff_slice.Sub() : &slope_coeffs_local;
mli_tensor* out_tsr = out_is_local ? out_slice.Sub() : &out_local;
slope_coeffs_tsr->el_params.fx.frac_bits = 7;

//MicroPrintf("slope_coeffs_tsr->data:%x",slope_coeffs_tsr->data);
  MicroPrintf("mli tensor of slope coeffs Q7 fixed point :%x",*(slope_coeffs->data.int8));

//slope_coeffs_tsr->data = (void*) &params->alpha;
//MicroPrintf("slope_coeffs_tsr->data:%f",slope_coeffs_tsr->data);


  //MicroPrintf("%x",*(slope_coeffs->data.f));

  while (!out_slice.Done()) {
    mli_mov_tensor_sync(input_slice.Sub(), &copy_config, input_tsr);
    mli_mov_tensor_sync(slope_coeff_slice.Sub(), &copy_config, slope_coeffs_tsr);

    mli_status res = mli_krn_leaky_relu_fx8(input_tsr, slope_coeffs_tsr, out_tsr);
    MicroPrintf("res:%d",res);
    mli_mov_tensor_sync(out_tsr, &copy_config, out_slice.Sub());
    input_slice.Next();
    slope_coeff_slice.Next();
    out_slice.Next();
  }

  return kTfLiteOk;
}
#endif
template <typename T>
void QuantizeLeakyRelu(const LeakyReluOpData& data,
                       const TfLiteEvalTensor* input,
                       TfLiteEvalTensor* output) {
  LeakyReluParams op_params = {};

  op_params.input_offset = data.input_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier_alpha = data.output_multiplier_alpha;
  op_params.output_shift_alpha = data.output_shift_alpha;
  op_params.output_multiplier_identity = data.output_multiplier_identity;
  op_params.output_shift_identity = data.output_shift_identity;

  reference_ops::QuantizeLeakyRelu(op_params,
                                   tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorData<T>(input),
                                   tflite::micro::GetTensorShape(output),
                                   tflite::micro::GetTensorData<T>(output));
}

void* LeakyReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(LeakyReluOpData));
}

TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const LeakyReluOpData& data = *static_cast<LeakyReluOpData*>(node->user_data);
  LeakyReluParams op_params = {};
  const auto* params =
      static_cast<TfLiteLeakyReluParams*>(node->builtin_data);
      TfLiteEvalTensor* slope_coeffs =
          tflite::micro::GetEvalInput(context, node, params->alpha);
//TfLiteEvalTensor slope_coeffs    ;
  //        slope_coeffs.data.data = (float *)&params->alpha;
    //      slope_coeffs.type = kTfLiteFloat32;
  //  MicroPrintf("%f",(slope_coeffs->data.int8));
  //  MicroPrintf("%f",*(slope_coeffs->data.int8));
  //  MicroPrintf("fixed 8: %x\n", slope_coeffs->data.data);
  //  MicroPrintf("fixed 8: %x\n", (int8_t*)&(slope_coeffs->data.int8));
  //  MicroPrintf("tensor->data.int8:%x",*(slope_coeffs->data.int8));
          MicroPrintf("Inside LeakyReluEval params->alpha:%f",params->alpha);
          int8_t val_i8 = (int8_t)(params->alpha*(1<<7));
          slope_coeffs->data.data = (int8_t *)&val_i8;
          slope_coeffs->type = kTfLiteInt8;
          MicroPrintf("Converted to Q7 fixed point tensor->data.int8:%x",*(slope_coeffs->data.int8));
          //MicroPrintf("%f",*(slope_coeffs.data.data));
  switch (input->type) {
    case kTfLiteFloat32: {


      op_params.alpha = params->alpha;
      reference_ops::LeakyRelu(op_params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
#ifdef TEST_ARC_MLI
      if (data.is_mli_applicable) {

        MicroPrintf("\n Entering EvalMLI\n");
        EvalMli(context, params, data, input,slope_coeffs, output);
        MicroPrintf("\n Exiting EvalMLI\n");
      }
#else
  QuantizeLeakyRelu<int8_t>(data, input, output);
#endif

      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      QuantizeLeakyRelu<int16_t>(data, input, output);
      return kTfLiteOk;
    } break;
    default:
      MicroPrintf("Only float32, int8 are supported by LEAKY_RELU, got %s.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteError;
}

TfLiteRegistration Register_LEAKY_RELU() {
  return {/*init=*/LeakyReluInit,
          /*free=*/nullptr,
          /*prepare=*/LeakyReluPrepare,
          /*invoke=*/LeakyReluEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
