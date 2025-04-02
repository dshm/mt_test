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


def evaluate_retinaface_dynamic_batchsize(model_path : str, xpu_type: str, batchsize: int):
    if xpu_type == 'musa': 
        test_xpu_session = ort.InferenceSession(model_path, providers=['MUSAExecutionProvider'])
    else:
        test_xpu_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    test_cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    inputs_info = test_cpu_session.get_inputs()
    outputs_info = test_cpu_session.get_outputs()

    output_names = []

    input_dict = {}
    input_data = np.random.randn(batchsize, 3, 640, 640).astype(np.float32)
    input_dict["input"] = input_data

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
    xpu_result = []
    total_time = 0.0

    print('len(cpu_result)')
    print(len(cpu_result))
    print('cpu_result[0].shape')
    print(cpu_result[0].shape)
    print('cpu_result[1].shape')
    print(cpu_result[1].shape)
    print('cpu_result[2].shape')
    print(cpu_result[2].shape)

    for i in range(warm_up):
        test_xpu_session.run(output_names, input_dict)

    for i in range(iter):
        start_time = time.time()
        xpu_result = test_xpu_session.run(output_names, input_dict)
        total_time += time.time() - start_time
    
    print('len(xpu_result)')
    print(len(xpu_result))
    print('xpu_result[0].shape')
    print(xpu_result[0].shape)
    print('xpu_result[1].shape')
    print(xpu_result[1].shape)
    print('xpu_result[2].shape')
    print(xpu_result[2].shape)

    max_difference = 0.0
    max_difference0 = 0.0
    max_difference1 = 0.0
    max_difference2= 0.0
    L2norm = 0.0
    L2norm0 = 0.0
    L2norm1 = 0.0
    L2norm2 = 0.0  

    max_difference0 = np.max(np.abs(xpu_result[0] - cpu_result[0]))
    max_difference1 = np.max(np.abs(xpu_result[1] - cpu_result[1]))
    max_difference2 = np.max(np.abs(xpu_result[2] - cpu_result[2]))
    # max_difference = max(max_difference0, max_difference1, max_difference2) 
    
    L2norm0 = np.sum(np.abs(xpu_result[0] - cpu_result[0])) / xpu_result[0].size
    L2norm1 = np.sum(np.abs(xpu_result[1] - cpu_result[1])) / xpu_result[1].size
    L2norm2 = np.sum(np.abs(xpu_result[2] - cpu_result[2])) / xpu_result[2].size
    # L2norm = L2norm0 + L2norm1 + L2norm2

    # print("Batch Size: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / batch".format(iter, total_time, 1000.0 * total_time / iter))
    print("Batch Size: {}\nSample Number: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / one_sample".format(batchsize, iter*batchsize, total_time, 1000.0 * total_time / (iter*batchsize)))
    print("Throughput rate: ", 1000.0/( 1000.0 * total_time / (iter*batchsize) )   )
    throughput_rate.append(1000.0/( 1000.0 * total_time / (iter*batchsize) ))

    # print(xpu_result[0].shape)
    return max_difference0, L2norm0, max_difference1, L2norm1, max_difference2, L2norm2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help='Specify model path of current network. ', required=True) 
    parser.add_argument('--xpu', type=str, default='xpu', help='Specify xpu type of current test task. The default device is CUDA. ') 
    args = parser.parse_args() 
    
    ort.set_default_logger_severity(3)

    batch_size_num=[1, 4, 8, 16, 32, 64]
    for bs_num in batch_size_num:
        print("Batch Size: ", bs_num)
        md0, l20, md1, l21, md2, l22 = evaluate_retinaface_dynamic_batchsize(args.model, args.xpu, bs_num)
        
        print('Max: output0: {}  output1: {}  output2: {}'.format(md0, md1, md2))
        print('Relative Difference: output0: {}  output1: {}  output2: {}'.format(l20, l21, l22))

    for idx in range(len(batch_size_num)):
        print("Batchsize ", batch_size_num[idx])
        print("Throughput rate ", throughput_rate[idx])