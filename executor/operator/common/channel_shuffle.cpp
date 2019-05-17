/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: guo.xinzhan@seiriosai.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/channel_shuffle.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace ShuffleChannelImpl {

struct ShuffleChannelOps : public NodeOps
{
    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        ShuffleChannel* ShuffleChannel_op = dynamic_cast<ShuffleChannel*>(node->GetOp());
        ShuffleChannelParam* param = ShuffleChannel_op->GetParam();

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();
        int group = param->group;
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        if(dims.size()==4){
            int batch_number = dims[0];
            int channel = dims[1];
            int hw_size = dims[2]*dims[3];
            int chs_per_group = channel / group; 
           
            if (channel != chs_per_group * group)
            {
                // reject invalid group
                return -100;
            }

            for (int b = 0; b < batch_number; b++)
            {
                for (int i = 0; i != group; i++)
                {
                    for (int j = 0; j != chs_per_group; j++)
                    {
                        int src_q = (chs_per_group * i + j) * hw_size;
                        int dst_q = (group * j + i) * hw_size;
                        memcpy(output + b * hw_size * channel + dst_q, input + b * hw_size * channel + src_q, hw_size*sizeof(float));
                    }
                }
            }
        }
        else if(dims.size()==3){
            int channel = dims[0];
            int hw_size = dims[1]*dims[2];
            int chs_per_group = channel / group; 
           
            if (channel != chs_per_group * group)
            {
                // reject invalid group
                return -100;
            }
         
            for (int i = 0; i != group; i++)
            {
                for (int j = 0; j != chs_per_group; j++)
                {
                    int src_q = (chs_per_group * i + j)*hw_size;
                    int dst_q = (group * j + i)*hw_size;
                    memcpy(output+dst_q, input+src_q, hw_size*sizeof(float));
                }
            }

        }

        return true;
    }
};
NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    ShuffleChannelOps* ops = new ShuffleChannelOps();

    return ops;
}

}    // namespace ShuffleChannelImpl

using namespace ShuffleChannelImpl;

void RegisterShuffleChannelNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "ShuffleChannel", ShuffleChannelImpl::SelectFunc, 1000);
}

}    // namespace TEngine
