import inspect
import os
import sys
import types
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import accelerate  # noqa: F401
except ModuleNotFoundError:
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.__spec__ = ModuleSpec("accelerate", loader=None)
    accelerate_stub.Accelerator = object
    sys.modules["accelerate"] = accelerate_stub

dataloader_stub = types.ModuleType("src.data.dataloader")
dataloader_stub.__spec__ = ModuleSpec("src.data.dataloader", loader=None)
dataloader_stub.SequenceDataset = object
sys.modules["src.data.dataloader"] = dataloader_stub

diy_dataloader_stub = types.ModuleType("src.data.dataloader_diy_data")
diy_dataloader_stub.__spec__ = ModuleSpec(
    "src.data.dataloader_diy_data", loader=None
)
diy_dataloader_stub.gumbel_softmax = lambda logits, **_kwargs: logits
sys.modules["src.data.dataloader_diy_data"] = diy_dataloader_stub

from src.utils.train_loop_basic import BasicTrainLoop
from src.utils.train_multi_gpu import TrainLoop_multi_gpu
from src.utils.sample_util import inference
from src.utils.train_single_gpu import TrainLoop_single_gpu


class _InitOnlyAccelerator:
    is_main_process = True
    device = torch.device("cpu")


class _SamplingAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def autocast():
        return nullcontext()


class _SamplingModel:
    cond_weight = 4.0

    def to(self, _device):
        return self

    def eval(self):
        return self


def test_legacy_constructor_keywords_and_order_are_preserved():
    multi = inspect.signature(TrainLoop_multi_gpu.__init__).parameters
    single = inspect.signature(TrainLoop_single_gpu.__init__).parameters

    assert multi["with_condition"].default is True
    assert list(multi).index("tgt_values") < list(multi).index("with_condition")
    assert list(multi).index("with_condition") < list(multi).index("label_names")

    assert single["do_gumbel_softmax"].default is False
    assert list(single).index("do_gumbel_softmax") < list(single).index("tgt_values")
    assert list(single).index("tgt_values") < list(single).index("label_names")

    for loop_class in (TrainLoop_multi_gpu, TrainLoop_single_gpu):
        loader_parameters = list(
            inspect.signature(loop_class._prepare_data_loader).parameters
        )
        assert loader_parameters == ["self", "data", "batch_size", "num_workers"]


def test_inference_keeps_the_legacy_positional_prefix():
    parameters = list(inspect.signature(inference).parameters)
    assert parameters[:9] == [
        "diffusion_model",
        "class_num",
        "cond_weight",
        "sample_bs",
        "seq_len",
        "output_all_steps",
        "target_values",
        "device",
        "with_condition",
    ]
    assert parameters[9:] == ["label_names", "fast_gen"]


def test_sampling_only_loops_keep_the_legacy_three_class_default():
    single = TrainLoop_single_gpu(
        data={},
        model=torch.nn.Linear(1, 1),
        accelerator=_InitOnlyAccelerator(),
        end_epoch=1,
    )
    multi = TrainLoop_multi_gpu(
        data={},
        model=torch.nn.Linear(1, 1),
        accelerator=_InitOnlyAccelerator(),
        end_epoch=1,
    )

    assert single.num_classes == 3
    assert multi.num_classes == 3


def test_multi_gpu_condition_switch_preserves_legacy_unconditional_training():
    loop = TrainLoop_multi_gpu.__new__(TrainLoop_multi_gpu)
    labels = object()

    loop.with_condition = True
    assert loop._condition_labels(labels) is labels
    loop.with_condition = False
    assert loop._condition_labels(labels) is None


def test_multi_gpu_sample_forwards_the_condition_switch():
    loop = TrainLoop_multi_gpu.__new__(TrainLoop_multi_gpu)
    loop.accelerator = _SamplingAccelerator()
    loop.ema_model = _SamplingModel()
    loop.seq_len = 50
    loop.num_classes = 3
    loop.label_names = None
    loop.tgt_values = None
    loop.with_condition = False
    loop.checkpoint_dir = None
    loop.save_name = "legacy"
    loop.end_epoch = 2

    with patch("src.utils.train_multi_gpu.inference", return_value=[">seq\nAAA\n"]) as inference:
        loop.sample(epoch=1, write_fasta=False)

    assert inference.call_args.kwargs["with_condition"] is False


def test_single_gpu_gumbel_softmax_parameters_and_disabled_identity():
    loop = TrainLoop_single_gpu.__new__(TrainLoop_single_gpu)
    value = object()

    loop.do_gumbel_softmax = False
    assert loop._prepare_input(value, training=True) is value

    loop.do_gumbel_softmax = True
    with patch("src.utils.train_single_gpu.gumbel_softmax", side_effect=("train", "valid")) as gumbel:
        assert loop._prepare_input(value, training=True) == "train"
        assert loop._prepare_input(value, training=False) == "valid"

    assert gumbel.call_args_list[0].args == (value,)
    assert gumbel.call_args_list[0].kwargs == {"scale": 3, "tau": 0.8, "hard": False}
    assert gumbel.call_args_list[1].args == (value,)
    assert gumbel.call_args_list[1].kwargs == {"scale": 1, "tau": 1, "hard": True}


def test_checkpoint_path_defaults_to_legacy_and_mcml_can_opt_in(tmp_path):
    loop = BasicTrainLoop.__new__(BasicTrainLoop)
    loop.save_name = "legacy_run"
    loop.checkpoint_dir = None
    assert loop._checkpoint_path(12) == os.path.join(
        "checkpoints", "legacy_run_at_12epoch.pt"
    )

    checkpoint_dir = tmp_path / "outputs" / "mcml" / "checkpoints"
    loop.checkpoint_dir = str(checkpoint_dir)
    expected = checkpoint_dir / "epoch_2000.pt"
    assert loop._checkpoint_path(2000) == str(expected)

    loop.accelerator = SimpleNamespace(get_state_dict=lambda model: {"state": model})
    loop.model = "model-state"
    loop.optimizer = SimpleNamespace(state_dict=lambda: {"optimizer": "state"})
    with patch("src.utils.train_loop_basic.torch.save") as save:
        loop.save_checkpoint(epoch=2000)

    assert checkpoint_dir.is_dir()
    assert save.call_args.args[1] == str(expected)
