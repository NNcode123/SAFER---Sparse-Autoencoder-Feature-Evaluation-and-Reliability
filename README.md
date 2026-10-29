# SAFER-Sparse-Autoencoder-Feature-Evaluation-and-Reliability
This research is dedicated to using Sparse Autoencoders to investigate how interpretability techniques can influence and audit the behavior of open-weight language models, including Gemma.

## Project structure

```text
scripts/                 Experiment and workflow scripts
models/                  Model-related code and assets
Figures/                 Generated figures
build_tools/
  Hipergator/            HiPerGator-specific build and run tools
  Local/                 Local build and run tools
utils/                   Shared utilities
```

## Interpretability workflow

The interpretability experiments will use several prompt datasets, cached as
dataframes in `datasets_dir`. Each dataset should include a `prompt` column
and a category column (for example, `type`) that labels the failure mode or
undesirable behavior the prompt is intended to elicit. Use
`save_or_load_df_at_url` in `scripts/py_scripts/data.py` to load each dataset
from its URL on first use and reuse the cached dataframe on subsequent runs.

For each dataset:

1. Preprocess the dataframe into a consistent format: retain the prompt and
   category columns, remove rows without prompts, trim prompt whitespace, and
   remove duplicate prompt/category pairs.
2. Select a small representative sample (about two prompts) from each failure
   mode or undesirable-behavior category.
3. Run the sampled prompts through the model while collecting the relevant
   Sparse Autoencoder (SAE) activations. Analyze which SAE features activate
   and how those activations relate to the behavior in the model's output.
4. Summarize the analysis in a table, including the dataset, category, sampled
   prompt, relevant model output, SAE features or activations, and the
   interpretation of their relationship to the elicited behavior.
