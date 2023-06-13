![ETH MTC HEADER](../../../assets/ETHMTCHeaderOriginal.png)

# HuggingFace Trainer Mixin's

### Token Batch Mixin

This trainer mixin allows for batching at a token level, instead of the
traditional sentence level batching. This is of particular help to optimize 
memory usage while working with sentences with variable tokenized lenghts 
(a lot a short sentences use the same amount of memory than a lot of long 
sentences). This can lead to huge speed-ups.

#### Standard BART

The files / classes needed for this mixin are:

1. `summarization/trainer/mixin/token_batch_mixin.py` :arrow_right: `TokenBatchMixin`
2. `summarization/dataset/sampler.py` :arrow_right: `TokenBatchSampler`
3. `utils/arguments.py` :arrow_right: `TokenBatchArguments`


The following things need to be made to integrate the mixin into another project:

- Extend the HuggingFace TrainerArguments:

```python
@dataclass
class CustomTrainingArguments(TokenBatchArguments, TrainingArguments):
    """ HuggingFace arguments for using the poissibility of token batching. """
    pass
```

- Extend the Huggingface Trainer:

```python
class CustomSeq2SeqTrainer(TokenBatchMixin, Seq2SeqTrainer):
    """ HuggingFace trainer extended with the poissibility of token batching. """
    pass
```

**Note**: Considers `input_ids` (and `label`) tokens with/without padding

#### Guided BART

There is also a version that can be used together with the GuidedBartModel that
takes into account the tokens in the guidance signal.

The files / classes needed for this mixin are:

1. `summarization/trainer/mixin/token_batch_mixin.py` :arrow_right: `GuidedTokenBatchMixin`
2. `summarization/dataset/sampler.py` :arrow_right: `TokenBatchSampler`
3. `utils/arguments.py` :arrow_right: `GuidedTokenBatchArguments`


Extend the Huggingface Trainer:

- Extend the HuggingFace TrainerArguments:

```python
@dataclass
class CustomTrainingArguments(GuidedTokenBatchArguments, TrainingArguments):
    """ HuggingFace arguments for using the poissibility of token batching. """
    pass
```

- Extend the Huggingface Trainer:

```python
class CustomSeq2SeqTrainer(GuidedTokenBatchMixin, Seq2SeqTrainer):
    """ HuggingFace trainer extended with the poissibility of token batching. """
    pass
```

**Note**: Considers `input_ids`, `guidance_ids` (and `label`) tokens with/without padding

### Train Metrics Mixin

This trainer mixin allows for tracking+logging of metrics on the train dataset.

The files / classes needed for this mixin are:


1. `summarization/trainer/mixin/train_metrics_mixin.py` :arrow_right: `TrainMetricsMixin`
2. `utils/arguments.py` :arrow_right: `TrainMetricsArguments`


The following things need to be made to integrate the mixin into another project:

- Extend the HuggingFace TrainerArguments:

```python
@dataclass
class CustomTrainingArguments(TrainMetricsArguments, TrainingArguments):
    """ HuggingFace arguments for using the poissibility of train metrics logging. """
    pass
```

- Extend the Huggingface Trainer:

```python
class CustomSeq2SeqTrainer(TrainMetricsMixin, Seq2SeqTrainer):
    """ HuggingFace trainer extended with the poissibility of train metrics logging. """
    pass
```