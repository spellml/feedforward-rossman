# feedforward-rossman

`pytorch` training script implementing a feedforward network on the [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition.

This is a tabular time-series dataset, traditionally the domain of gradient boosted tree libraries like `xgboost`. However, recent advances in generalized embeddings (mostly stemming from the NLP world) have put feedforward neural networks on par with GBTs in terms of performance.

This `pytorch` implementation adapts the `fastai` model presented in [Lesson 3](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) of the FastAI course.

To run on Spell (requires being in the `spellrun` org for the GH integration):

```python
prodspell run \
  --machine-type V100 \
  --github-url https://github.com/ResidentMario/spell-feedforward-rossman.git \
  --pip kaggle \
  "chmod +x /spell/scripts/download_data.sh; /spell/scripts/download_data.sh; python /spell/models/model_1.py"
```
