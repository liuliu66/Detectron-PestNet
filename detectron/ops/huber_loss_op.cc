/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "huber_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(HuberLoss, HuberLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    HuberLossGradient,
    HuberLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(HuberLoss)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Custom Defined Huber Loss
)DOC")
    .Arg(
        "tau",
        "(float) default 1.0; L2 to L1 transition point.")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Input(
        0,
        "Y_hat",
        "Tensor of predictions (at least 1D).")
    .Input(
        1,
        "Y",
        "Tensor of labels with the same shape as Y_hat.")
    .Input(
        2,
        "alpha_in",
        "Tensor of inside weights with the same shape as Y.")
    .Input(
        3,
        "alpha_out",
        "Tensor of outside weights with the same shape as Y.")
    .Output(
        0,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(HuberLossGradient)
    .NumInputs(5)
    .NumOutputs(1)
    .Input(
        0,
        "Y_hat",
        "See HuberLoss.")
    .Input(
        1,
        "Y",
        "See HuberLoss.")
    .Input(
        2,
        "alpha_in",
        "See HuberLoss.")
    .Input(
        3,
        "alpha_out",
        "See HuberLoss.")
    .Input(
        4,
        "d_loss",
        "Gradient of forward output 0 (loss).")
    .Output(
        0,
        "d_Y_hat",
        "Gradient of forward input 0 (Y_hat).");

class GetHuberLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HuberLossGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(HuberLoss, GetHuberLossGradient);

} // namespace caffe2
