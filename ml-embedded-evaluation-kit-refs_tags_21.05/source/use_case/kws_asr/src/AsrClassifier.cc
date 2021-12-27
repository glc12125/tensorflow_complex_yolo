/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "AsrClassifier.hpp"

#include "hal.h"
#include "TensorFlowLiteMicro.hpp"
#include "Wav2LetterModel.hpp"

template<typename T>
bool arm::app::AsrClassifier::GetTopResults(TfLiteTensor* tensor,
                                            std::vector<ClassificationResult>& vecResults,
                                            const std::vector <std::string>& labels, double scale, double zeroPoint)
{
    const uint32_t nElems = tensor->dims->data[arm::app::Wav2LetterModel::ms_outputRowsIdx];
    const uint32_t nLetters = tensor->dims->data[arm::app::Wav2LetterModel::ms_outputColsIdx];

    if (nLetters != labels.size()) {
        printf("Output size doesn't match the labels' size\n");
        return false;
    }

    /* NOTE: tensor's size verification against labels should be
     *       checked by the calling/public function. */
    if (nLetters < 1) {
        return false;
    }

    /* Final results' container. */
    vecResults = std::vector<ClassificationResult>(nElems);

    T* tensorData = tflite::GetTensorData<T>(tensor);

    /* Get the top 1 results. */
    for (uint32_t i = 0, row = 0; i < nElems; ++i, row+=nLetters) {
        std::pair<T, uint32_t> top_1 = std::make_pair(tensorData[row], 0);

        for (uint32_t j = 1; j < nLetters; ++j) {
            if (top_1.first < tensorData[row + j]) {
                top_1.first = tensorData[row + j];
                top_1.second = j;
            }
        }

        double score = static_cast<int> (top_1.first);
        vecResults[i].m_normalisedVal = scale * (score - zeroPoint);
        vecResults[i].m_label = labels[top_1.second];
        vecResults[i].m_labelIdx = top_1.second;
    }

    return true;
}
template bool arm::app::AsrClassifier::GetTopResults<uint8_t>(TfLiteTensor* tensor,
                                                              std::vector<ClassificationResult>& vecResults,
                                                              const std::vector <std::string>& labels, double scale, double zeroPoint);
template bool arm::app::AsrClassifier::GetTopResults<int8_t>(TfLiteTensor* tensor,
                                                             std::vector<ClassificationResult>& vecResults,
                                                             const std::vector <std::string>& labels, double scale, double zeroPoint);

bool arm::app::AsrClassifier::GetClassificationResults(
            TfLiteTensor* outputTensor,
            std::vector<ClassificationResult>& vecResults,
            const std::vector <std::string>& labels, uint32_t topNCount)
{
        vecResults.clear();

        constexpr int minTensorDims = static_cast<int>(
            (arm::app::Wav2LetterModel::ms_outputRowsIdx > arm::app::Wav2LetterModel::ms_outputColsIdx)?
             arm::app::Wav2LetterModel::ms_outputRowsIdx : arm::app::Wav2LetterModel::ms_outputColsIdx);

        constexpr uint32_t outColsIdx = arm::app::Wav2LetterModel::ms_outputColsIdx;

        /* Sanity checks. */
        if (outputTensor == nullptr) {
            printf_err("Output vector is null pointer.\n");
            return false;
        } else if (outputTensor->dims->size < minTensorDims) {
            printf_err("Output tensor expected to be 3D (1, m, n)\n");
            return false;
        } else if (static_cast<uint32_t>(outputTensor->dims->data[outColsIdx]) < topNCount) {
            printf_err("Output vectors are smaller than %" PRIu32 "\n", topNCount);
            return false;
        } else if (static_cast<uint32_t>(outputTensor->dims->data[outColsIdx]) != labels.size()) {
            printf("Output size doesn't match the labels' size\n");
            return false;
        }

        if (topNCount != 1) {
            warn("TopNCount value ignored in this implementation\n");
        }

        /* To return the floating point values, we need quantization parameters. */
        QuantParams quantParams = GetTensorQuantParams(outputTensor);

        bool resultState;

        switch (outputTensor->type) {
            case kTfLiteUInt8:
                resultState = this->GetTopResults<uint8_t>(
                        outputTensor, vecResults,
                        labels, quantParams.scale,
                        quantParams.offset);
                break;
            case kTfLiteInt8:
                resultState = this->GetTopResults<int8_t>(
                        outputTensor, vecResults,
                        labels, quantParams.scale,
                        quantParams.offset);
                break;
            default:
                printf_err("Tensor type %s not supported by classifier\n",
                    TfLiteTypeGetName(outputTensor->type));
                return false;
        }

        if (!resultState) {
            printf_err("Failed to get sorted set\n");
            return false;
        }

        return true;
}