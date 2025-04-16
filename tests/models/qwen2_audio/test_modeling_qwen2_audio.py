# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import inspect

import numpy as np
import pytest
import torch
from transformers import Qwen2AudioConfig, Qwen2Config, Qwen2AudioEncoderConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.models.modeling_common import ids_numpy, floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [0, 1]


class Qwen2AudioTester:
    def __init__(
        self,
        batch_size=2,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=32,
        max_position_embeddings=512,
        use_cache=False,
        output_attentions=False,
        attn_implementation="eager",
        pad_token_id=0,
        num_mel_bins=32,
        encoder_layers=2,
        encoder_attention_heads=8,
        encoder_ffn_dim=32,
        d_model=128,
        max_source_positions=512,
    ):
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id

        self.num_mel_bins=num_mel_bins
        self.encoder_layers=encoder_layers
        self.encoder_attention_heads=encoder_attention_heads
        self.encoder_ffn_dim=encoder_ffn_dim
        self.d_model=d_model
        self.max_source_positions=max_source_positions


        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.attn_implementation = attn_implementation

    def get_large_model_config(self):
        return Qwen2AudioConfig.from_pretrained("BAAI/Emu3-Chat-hf")

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy(
            [self.batch_size, self.encoder_seq_length], self.vocab_size
        )
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy(
                [self.batch_size, self.encoder_seq_length], vocab_size=2
            )

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_numpy(
                [self.batch_size, self.decoder_seq_length], self.vocab_size
            )

        config = self.get_config()

        input_features = floats_numpy(
            [self.batch_size, self.num_mel_bins, self.max_source_positions]
        )
        max_seq_len = (self.max_source_positions - 2) // 2 + 1
        audio_attention_mask = ids_numpy(
            [self.batch_size,  1, max_seq_len, max_seq_len], vocab_size=2
        )

        return (
            config,
            input_ids,
            attention_mask,
            lm_labels,
            input_features,
            audio_attention_mask,
        )

    def get_config(self):
        text_config = Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            attn_implementation=self.attn_implementation,
        )
        audio_config = Qwen2AudioEncoderConfig(
            num_mel_bins=self.num_mel_bins,
            encoder_layers=self.encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            d_model=self.d_model,
            max_source_positions=self.max_source_positions,
        )
        config = Qwen2AudioConfig(audio_config, text_config)
        return config


model_tester = Qwen2AudioTester()
(
    config,
    input_ids,
    attention_mask,
    lm_labels,
    input_features,
    audio_attention_mask,
) = model_tester.prepare_config_and_inputs()


QWEN2AUDIO_CASES = [
    [
        "Qwen2Model",
        "transformers.Qwen2Model",
        "mindway.transformers.Qwen2Model",
        (config.text_config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "Qwen2AudioEncoder",
        "transformers.Qwen2AudioEncoder",
        "mindway.transformers.Qwen2AudioEncoder",
        (config.audio_config,),
        {},
        (),
        {"input_features": input_features, "attention_mask": audio_attention_mask, "return_dict": True},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in QWEN2AUDIO_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    (
        pt_inputs_args,
        pt_inputs_kwargs,
        ms_inputs_args,
        ms_inputs_kwargs,
    ) = generalized_parse_args(pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs)

    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    # print("ms:", ms_outputs)
    # print("pt:", pt_outputs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            # print("===map", pt_key, ms_idx)
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = getattr(ms_outputs, ms_idx)
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"{name}: ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
