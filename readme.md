# Self-Attention Attribution

This repository contains the implementation for AAAI-2021 paper [Self-Attention Attribution: Interpreting Information Interactions Inside Transformer](https://arxiv.org/pdf/2004.11207.pdf). It includes the code for generating the self-attention attribution score, pruning attention heads with our method, constructing the attribution tree and extracting the adversarial triggers. All of our experiments are conducted on bert-base-cased model, our methods can also be easily transfered to other Transformer-based models.

## Requirements
```bash
pip install -r requirements.txt
```

You can install attattr from source:
```bash
git clone https://github.com/YRdddream/attattr
cd attattr
pip install -e .
```

## Download Pre-Finetuned Models and Datasets

Before running self-attention attribution, you can first fine-tune bert-base-cased model on a downstream task (such as MNLI) by running the file `run_classifier_orig.py`.
We also provide the example datasets and the pre-finetuned checkpoints at [Google Drive](https://drive.google.com/file/d/1L8iI2wWSF6aVtYsJ9BtdQmEGbpnpE3dW/view?usp=sharing).

## Get Self-Attention Attribution Scores
Run the following command to get the self-attention attribution score and the self-attention score.
```bash
python src/attattr/generate_attrscore.py --task_name mnli --data_dir ./model_and_data/mnli_data \
       --bert_model bert-base-cased --batch_size 16 --num_batch 4 \
       --model_file ./model_and_data/model.mnli.bin --example_index 5 \
       --get_att_attr --get_att_score --output_dir ./output
```

## Construction of Attribution Tree
When you get the self-attribution scores of a target example, you could construct the attribution tree.
We recommend you to run the file `get_tokens_and_pred.py` to summarize the data
```bash
python src/attattr/get_tokens_and_pred.py --task_name mnli --data_dir ./model_and_data/mnli_data \
       --bert_model bert-base-cased \
       --model_file ./model_and_data/model.mnli.bin \
       --output_dir ./output
```

or you can manually change the value of `tokens` in `attribution_tree.py`.
```bash
python src/attattr/attribution_tree.py --attr_file ${attr_file} \
       --tokens_file ${tokens_file} \
       --task_name ${task_name} --example_index ${example_index} \
       --output_file
```

You can generate the attribution tree from the provided example.
```bash
python src/attattr/attribution_tree.py --attr_file ./output/attr_zero_base_exp5.json \
       --tokens_file ./output/tokens_and_pred_100.json \
       --task_name mnli --example_index 5 \
       --output_file tree.png
```

## Reference
If you find this repository useful for your work, you can cite the paper:

```
@inproceedings{attattr,
  author = {Yaru Hao and Li Dong and Furu Wei and Ke Xu},
  title = {Self-Attention Attribution: Interpreting Information Interactions Inside Transformer},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence},
  publisher = {{AAAI} Press},
  year      = {2021},
  url       = {https://arxiv.org/pdf/2004.11207.pdf}
}
```
