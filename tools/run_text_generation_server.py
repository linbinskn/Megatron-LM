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
import time

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
    group.add_argument("--kv_cache_quant", type=bool, default=False,
                       help='Size of the output generated text.')
    return parser


def benchmark(model, batch_size, output_seq):
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # prompts = ["Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's conflict with Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and Muggles (non-magical people). people people people people people people people people people people people people people people people"] * batch_size
    prompts = ["Harry " * 511] * batch_size
    tokens_to_generate = output_seq
    for i in range(1):
        torch.cuda.synchronize()
        generate_and_post_process(model, prompts, tokens_to_generate, use_eod_token_for_early_termination=False)
        torch.cuda.synchronize()

    start = time.time()
    for i in range(1):
        torch.cuda.synchronize()
        generate_and_post_process(model, prompts, tokens_to_generate, use_eod_token_for_early_termination=False)
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / 1
    print(f"batch_size: {batch_size}, output_seq: {output_seq}, elapsed time: {elapsed_ms}")
if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    model = get_model(model_provider, wrap_with_ddp=False)

    # import ipdb; ipdb.set_trace()
    benchmark(model, 1, 1536)