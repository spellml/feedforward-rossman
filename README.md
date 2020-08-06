# feedforward-rossman <a href="https://web.spell.ml/workspace_create?workspaceName=feedforward-rossman&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Ffeedforward-rossman&pip=kaggle&envVars=KAGGLE_USERNAME%3DYOUR_USERNAME,KAGGLE_KEY=YOUR_KEY"><img src=https://spell.ml/badge.svg height=20px/></a>
`pytorch` training script implementing a feedforward network on the [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition.

This is a tabular time-series dataset, traditionally the domain of gradient boosted tree libraries like `xgboost`. However, recent advances in generalized embeddings (mostly stemming from the NLP world) have put feedforward neural networks on par with GBTs in terms of performance.

This `pytorch` implementation adapts the `fastai` model presented in [Lesson 3](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) of the FastAI course.

To run code and notebooks in a Spell workspace:

```bash
spell jupyter --lab \
  --github-url https://github.com/spellml/feedforward-rossman.git \
  --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  feedforward-rossman
```

```bash
spell run \
  --machine-type V100 \
  --github-url https://github.com/spellml/feedforward-rossman.git \
  --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  "chmod +x /spell/scripts/download_data.sh /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_4.py"
```

```bash
spell run \
  --machine-type V100 \
  --github-url https://github.com/spellml/feedforward-rossman.git \
  --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  "chmod +x /spell/scripts/download_data.sh /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_4.py"
```
