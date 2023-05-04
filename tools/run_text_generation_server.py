# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
torch.manual_seed(0)
import time

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    #print("cu_prof_start")
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)

def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    #print("cu_prof_stop")
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_len):
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")
        
        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1] ## make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                    new_inference_key_memory, new_inference_value_memory)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    return parser

def benchmark_cudagraph(model, batch_size, output_seq):
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    
    tokens2use = torch.randint(1,500,(1,1)).cuda()
    positions2use = torch.randint(1,500,(1,1)).cuda()
    attention_mask = torch.randint(0,2,(1,1,1,1)).bool().cuda()
    num_attention_heads_per_partition = 12
    hidden_size_per_attention_head = 128
    
    inference_params = InferenceParams(batch_size,512+output_seq)
    for i in range(97):
        inference_key_memory = torch.empty(512+output_seq,batch_size,num_attention_heads_per_partition,hidden_size_per_attention_head,dtype=torch.float16,device=torch.cuda.current_device())
        inference_value_memory = torch.empty(512+output_seq,batch_size,num_attention_heads_per_partition,hidden_size_per_attention_head,dtype=torch.float16,device=torch.cuda.current_device())
        inference_params.key_value_memory_dict[i] = (inference_key_memory, inference_value_memory)

    output_tensor = model(tokens2use, positions2use, attention_mask, inference_params=inference_params)
    
    start = time.time()
    cu_prof_start()
    for i in range(16):
        torch.cuda.synchronize()
        output_tensor = model(tokens2use, positions2use, attention_mask, inference_params=inference_params)
        torch.cuda.synchronize()
    cu_prof_stop()
    elapsed_ms = (time.time() - start) * 1000 / 1
    print(f"batch_size: {batch_size}, output_seq: {output_seq}, elapsed time: {elapsed_ms}")

    print("pass")

def benchmark_cudagraph_on(model, batch_size, output_seq):
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    
    tokens2use = torch.randint(1,500,(1,1)).cuda()
    positions2use = torch.randint(1,500,(1,1)).cuda()
    attention_mask = torch.randint(0,2,(1,1,1,1)).bool().cuda()
    num_attention_heads_per_partition = 12
    hidden_size_per_attention_head = 128
    
    inference_params = InferenceParams(batch_size,512+output_seq)
    for i in range(97):
        inference_key_memory = torch.empty(512+output_seq,batch_size,num_attention_heads_per_partition,hidden_size_per_attention_head,dtype=torch.float16,device=torch.cuda.current_device())
        inference_value_memory = torch.empty(512+output_seq,batch_size,num_attention_heads_per_partition,hidden_size_per_attention_head,dtype=torch.float16,device=torch.cuda.current_device())
        inference_params.key_value_memory_dict[i] = (inference_key_memory, inference_value_memory)

    output_tensor = model(tokens2use, positions2use, attention_mask, inference_params=inference_params)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            output_tensor = model(tokens2use, positions2use, attention_mask, inference_params=inference_params)
    torch.cuda.current_stream().wait_stream(s)
    
    g = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(g):
            output_tensor = model(tokens2use, positions2use, attention_mask, inference_params=inference_params)
    
    start = time.time()
    torch.cuda.synchronize()
    cu_prof_start()
    for i in range(16):
        g.replay()
    torch.cuda.synchronize()
    cu_prof_stop()
    elapsed_ms = (time.time() - start) * 1000 / 1
    print(f"batch_size: {batch_size}, output_seq: {output_seq}, elapsed time: {elapsed_ms}")

def benchmark(model, batch_size, output_seq):
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    
    # input = torch.randint(1,500,(1,1))

    # prompts = ["Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's conflict with Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and Muggles (non-magical people). people people people people people people people people people people people people people people people"] * batch_size
    prompts = ["Harry " * 511] * batch_size
    tokens_to_generate = output_seq
    for i in range(10):
        torch.cuda.synchronize()
        output = generate_and_post_process(model, prompts, tokens_to_generate, use_eod_token_for_early_termination=False)
        torch.cuda.synchronize()

    start = time.time()
    for i in range(1):
        torch.cuda.synchronize()
        cu_prof_start()
        output = generate_and_post_process(model, prompts, tokens_to_generate, use_eod_token_for_early_termination=False)
        torch.cuda.synchronize()
        cu_prof_stop()
    elapsed_ms = (time.time() - start) * 1000 / 1
    print(f"batch_size: {batch_size}, output_seq: {output_seq}, elapsed time: {elapsed_ms}")
    torch.save(output, "o_tensor")
    
if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    model = get_model(model_provider, wrap_with_ddp=False)

    # import ipdb; ipdb.set_trace()

    benchmark(model, 1, 16)
    # benchmark_cudagraph_on(model, 1, 16)