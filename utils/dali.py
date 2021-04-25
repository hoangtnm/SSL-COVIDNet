import math

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def mux(condition, true_case, false_case):
    # See details at
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/expressions/expr_conditional_and_masking.html
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case


class DALIGenericIteratorV2(DALIGenericIterator):
    """Custom wrapper around DALIGenericIterator.

    See details at
    https://github.com/NVIDIA/DALI/blob/master/dali/python/nvidia/dali/plugin/base_iterator.py#L385
    """

    def __init__(self, pipelines, dataset_size: int, batch_size: int, *args, **kwargs):
        super().__init__(pipelines, *args, **kwargs)
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __len__(self):
        if self._last_batch_policy != LastBatchPolicy.DROP:
            return math.ceil(
                self.dataset_size / (self._num_gpus * self.batch_size))
        else:
            return self.dataset_size // (self._num_gpus * self.batch_size)
