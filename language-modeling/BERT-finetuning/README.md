# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
This repository contains the finetuing code for "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)" 
paper by [Jacob Devlin et al.](https://github.com/google-research/bert) on GLUE Dataset.

## Introuction
BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike the earlier language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models.
## Finetuning
### Prepare dataset
Download GLUE dataset by running this script:
```
python external/download_glue.py
```
It will download and extract the GLUE dataset in the current working directory. 

### Pre-trained Weights
Pre-trained *bert-base-uncased* weights, converted from [author's weights](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/language-modeling/BERT-finetuning/nbla_bert_params.h5). You can also refer to the [conversion code](https://github.com/sony/nnabla-examples/language-modeling/BERT-finetuning/convert_tf_params_to_nnabla.py) in case you'd like to know how tensorflow weights are converted to nnabla.

### Finetune and Evaluate
Run the following command and the evaluation commands in the table to finetune and evaluate the model on each task:

```shell
export GLUE_DIR=/path/to/glue_data
```

|[Task](https://gluebenchmark.com/tasks)|Metric|Score|Command|
|---|:---:|:---:|:---:|
|[CoLA](https://nyu-mll.github.io/CoLA/) |Matthew's corr|0.5624| ```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name $TASK_NAME     --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 64 --eval_batch_size=32 --train_batch_size=16   --num_train_epochs 3 --learning_rate 1e-4  --output_dir ./tmp/eval/$TASK_NAME``` |
|[SST-2](https://nlp.stanford.edu/sentiment/index.html) |Accuracy|0.9143| ```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name SST-2     --do_lower_case --data_dir $GLUE_DIR/SST-2 --max_seq_length 128 --eval_batch_size=32 --train_batch_size=16   --num_train_epochs 3 --learning_rate 2e-5  --output_dir ./tmp/eval/SST-2```|
|[MRPC](https://microsoft.com/en-us/download/details.aspx?id=52398) |Accuracy/F1|0.8732/0.9097|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name MRPC     --do_lower_case --data_dir $GLUE_DIR/MRPC --max_seq_length 128 --eval_batch_size=32 --train_batch_size=32   --num_train_epochs 3 --learning_rate 2e-5  --output_dir ./tmp/eval/MRPC```|
|[STS-B	](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) |Person/Spearman corr.|0.8804/0.8784|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name STS-B --do_lower_case --data_dir $GLUE_DIR/STS-B --max_seq_length 128 --eval_batch_size=32 --train_batch_size=32   --num_train_epochs 3 --learning_rate 3e-5  --output_dir ./tmp/eval/STS-B```|
|[QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) |Accuracy/F1|0.8725/0.8324|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name QQP     --do_lower_case --data_dir $GLUE_DIR/QQP --max_seq_length 128 --eval_batch_size=32 --train_batch_size=32   --num_train_epochs 3 --learning_rate 2e-5 --solver Adam  --output_dir ./tmp/eval/QQP```|
|[MNLI](http://www.nyu.edu/projects/bowman/multinli/) |Matched acc./Mismatched acc.	|0.8269/0.8392|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name MNLI    --do_lower_case --data_dir $GLUE_DIR/MNLI --max_seq_length 128 --eval_batch_size=32 --train_batch_size=32   --num_train_epochs 3 --learning_rate 2e-5 --solver Adam  --output_dir ./tmp/eval/MNLI```|
|[QNLI](https://rajpurkar.github.io/SQuAD-explorer/) |Accuracy|0.8794|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name QNLI    --do_lower_case --data_dir $GLUE_DIR/QNLI --max_seq_length 128 --eval_batch_size=32 --train_batch_size=16   --num_train_epochs 3 --learning_rate 2e-5  --output_dir ./tmp/eval/QNLI```|
|[RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) |Accuracy	|0.6642| ```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name RTE     --do_lower_case --data_dir $GLUE_DIR/RTE --max_seq_length 128 --eval_batch_size=32 --train_batch_size=32   --num_train_epochs 5 --learning_rate 2e-5  --output_dir ./tmp/eval/RTE```|
|[WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) |Accuracy|0.5742|```python finetune_and_eval.py --model_name_or_path bert-base-uncased --task_name WNLI    --do_lower_case --data_dir $GLUE_DIR/WNLI --max_seq_length 128 --eval_batch_size=32 --train_batch_size=16   --num_train_epochs 1 --learning_rate 1e-5  --output_dir ./tmp/eval/WNLI```|