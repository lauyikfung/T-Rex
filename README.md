# T-Rex Model

### Stage 0: Necessary Package Installation

+ pip install torchdrug
+ pip install wandb
+ pip install torch
+ pip install transformers
+ pip install openai
+ pip install rdkit
+ ```
  cd ChemicalReaction
  ```

### Stage 1: Candidate Ranking

+ First modify the "YOUR_WANDB_KEY" in scripts/train/candidate_generation.sh
+ Then run by the following code (you can change the parameters of fold and device):
  + ```
    bash scripts/train/candidate_ranking.sh
    ```
  + or the complete code(change FOLD, DEVICE NAME to the actual fold, GPU number and project name):
  + ```
    bash scripts/train/candidate_ranking.sh FOLD DEVICE NAME
    ```
+ Test by the following code
  + ```
    bash scripts/train/cr_test.sh
    ```
  + samely the complete code is: (due to the bug in torchdrug, the evaluation is shown with "Evaluate on train" but actually it is testing on some file)
  + ```
    bash scripts/train/cr_test.sh FOLD DEVICE NAME
    ```

### Mid-Stage: Candidate Preparation

+ You should have the openai key for generating the text description of the retrosynthesis pair.
+ First generate candidate pairs by:
  + ```
    bash scripts/chatgpt/candidate_generator.sh
    ```
+ Then generate the text descriptions using chatgpt("gpt-3.5-turbo-0301")
  + ```
    bash scripts/chatgpt/chatgpt_generator.sh
    ```

### Stage 2: Re-ranking

+ Then train re-rank model by:
  + ```
    bash scripts/train/reranking_training.sh
    ```
  + or the complete code(change FOLD, DEVICE NAME to the actual fold, GPU number and project name):
  + ```
    bash scripts/train/reranking_training.sh FOLD DEVICE NAME
    ```
+ And test by:
  + ```
    bash scripts/train/reranking_test.sh
    ```
  + or the complete code(change FOLD, DEVICE NAME to the actual fold, GPU number and checkpoint name, which is a little different from the NAME above, it is in logs-ckpt folder):
  + ```
    bash scripts/train/reranking_test.sh FOLD DEVICE NAME
    ```
