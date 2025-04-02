import torch
import onnxruntime as ort
import numpy as np
import random
import sys
import time

import argparse

throughput_rate=[]

ort_type_to_numpy_type_map = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": bool,
        }


def evaluate_arcface_dynamic_batchsize(model_path : str, xpu_type: str, batchsize: int):
    if xpu_type == 'musa': 
        test_xpu_session = ort.InferenceSession(model_path, providers=['MUSAExecutionProvider'])
    else:
        test_xpu_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    test_cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    inputs_info = test_cpu_session.get_inputs()
    outputs_info = test_cpu_session.get_outputs()

    output_names = []

    input_dict = {}
    input_data = np.random.randn(batchsize, 3, 112, 112).astype(np.float32)
    input_dict["data"] = input_data

    # for input in inputs_info:
    #     print(input.shape)
    #     print(input.type)
    #     input_data = np.random.randn(*input.shape).astype(ort_type_to_numpy_type_map[input.type])
    #     input_dict[input.name] = input_data

    for output in outputs_info:
        output_names.append(output.name)

    warm_up = 10
    iter = 10


    cpu_result = test_cpu_session.run(output_names, input_dict)
    # print('cpu_result shape')
    # print(len(cpu_result))
    # print(cpu_result)
    # print(cpu_result[0])
    print(cpu_result[0].shape)
    xpu_result = []
    total_time = 0.0

    for i in range(warm_up):
        test_xpu_session.run(output_names, input_dict)


    print('output_names')
    print(output_names)

    for i in range(iter):
        start_time = time.time()
        xpu_result = test_xpu_session.run(output_names, input_dict)
        total_time += time.time() - start_time
    
    # print('xpu_result shape')
    # print(len(xpu_result))
    # print(xpu_result)
    # print(xpu_result[0])
    print(xpu_result[0].shape)

    # print('output_names')
    # print(output_names)


    max_difference = 0.0
    L2norm = 0.0

    max_difference = np.max(np.abs(xpu_result[0] - cpu_result[0]))
    
    L2norm = np.sum(np.abs(xpu_result[0] - cpu_result[0])) / xpu_result[0].size

    print("Batch Size: {}\nSample Number: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / one_sample".format(batchsize, iter*batchsize, total_time, 1000.0 * total_time / (iter*batchsize)))
    print("Throughput rate: ", 1000.0/( 1000.0 * total_time / (iter*batchsize) )   )
    throughput_rate.append(1000.0/( 1000.0 * total_time / (iter*batchsize) ))

    return max_difference, L2norm

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help='Specify model path of current network. ', required=True) 
    parser.add_argument('--xpu', type=str, default='cuda', help='Specify xpu type of current test task. The default device is CUDA. ') 
    args = parser.parse_args()

    ort.set_default_logger_severity(3) 
    
    batch_size_num=[1, 4, 8, 16, 32, 64]
    for bs_num in batch_size_num:
        print("Batch Size: ", bs_num)
        md, l2 = evaluate_arcface_dynamic_batchsize(args.model, args.xpu, bs_num)
    
        print("Max: ", md)
        print("Relative Difference: ", l2)


    for idx in range(len(batch_size_num)):
        print("Batchsize ", batch_size_num[idx])
        print("Throughput rate ", throughput_rate[idx])

    # md, l2 = evaluate_arcface_dynamic_batchsize(args.model, args.xpu, 4)
    
    # print("Max: ", md)
    # print("Relative Difference: ", l2)

    # print('Device: {}\ndata type: fp16\ndataset size: {}\nrequired top1: 78.00%, top1: {:.2f}%\nbatch size is 24\nuse time: {:.2f} Seconds\nlatency: {:.2f}ms/batch\nthroughput: {:.2f} fps'.format(gpu_id, dataset_size, top1_accuracy.item(), total_time, 1000.0 * total_time / batch_cnt, batch_cnt * 24 / total_time))