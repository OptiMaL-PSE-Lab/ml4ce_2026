# ML4CE-RL: Reinforcement learning for a home energy microgrid

You will code an optimization algorithm to derive efficient control policies for the heat pump and EV battery of five households, so that together they reduce the electricity consumption while maintaining comfortable indoor conditions and preventing battery wear. 

## 1. Start the coursework

Make sure to have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed and you're using the latest version of the coursework. Then set the folder corresponding to this coursework as the current working directory.

```bash
git clone https://github.com/OptiMaL-PSE-Lab/ml4ce_2026.git
cd ml4ce_2026/RL_coursework
```

Create the required Python environment. The following command installs Python 3.12 and every package in the exact versions used for grading into a local `.venv/` folder.  
```bash
uv sync --locked
```
### Working in JupyterLab
To open JupyterLab in your browser, run: 
```bash
uv run jupyter lab
```
If it doesn't open automatically, copy the `http://localhost:8888/lab?token=...` link it prints.

Open `RL_coursework.ipynb`.

Stop Jupyter anytime with `Ctrl+C` and resume later with `uv run jupyter lab`: your files are on disk. Only **variables in memory** are lost: stopping Jupyter or restarting the kernel resets anything you haven't saved to a file. 

### Working in VSCode (recommended) 
Open the `RL_coursework/` folder and select the `.venv` interpreter as the notebook kernel.

## 2. Folder content
| Item | What it contains |
|------|------------------|
|`README.md` | This file; basic setup and instructions |
|`RL_coursework.ipynb` | The notebook you will work with|
|`algorithms/student_algorithm.py` | Template file for your implementation |
|`algorithms/cem.py` | Implementation of Cross-Entropy Method (use as reference) |
|`algorithms/reinforce_baseline.py` | Implementation of REINFORCE with baseline (use as reference)|
| `course_package` | Internals of the coursework, defining environment dynamics, conventional controllers and feature engineering pipeline. |
|`data/datasets/training/` | Training data; do not modify it | 
|`outputs/` | Folder created automatically for trained policies, plots and submissions |  

## 3. Your workflow

| Step | What to do | Where |
|---|---|---|
| **1. Explore** | Run the cells that create the environment, reset it and take a few actions. | Notebook, section 1 |
| **2. Compare** | Train the provided **CEM** and **REINFORCE** over the curriculum and read their transfer matrices. | Notebook, section 3.1 |
| **3. Write yours** | Define your shared policy, `AlgorithmConfig` and `train()` in the blank cell. Running the cell saves it as `algorithms/student_algorithm.py`. | Notebook, section 3.2 |
| **4. Test** | Run the curriculum with your algorithm and inspect its returns. Iterate. | Notebook, section 3.2 |
| **5. Save** | Set your group id in the last cell and run it. It writes `outputs/submissions/<your group>.py` and checks that `train(env, config, seed, policy=None)` and `AlgorithmConfig` exist. | Notebook, section 3.3 |

Hand in that one file: `outputs/submissions/<your_group>.py` (with your own group id). It must be self-contained, because it is run on its own: see [Section 3](#3-what-you-write).

## 3. What you write

Your submission is **one Python file** that contains **everything** you are asked to define (RL policy definition and optimization algorithm). The environment and its observation/action format are provided by the course and cannot be altered.

```python
class SharedPolicy:  # any name, any representation
    def act(self, local_obs):  # ONE agent's observation -> that agent's action
        ...  # (num_features,) -> (num_actions,), values in [-1, 1]


@dataclass
class AlgorithmConfig(
    BaseAlgorithmConfig
):  # your hyperparameters, all with default values
    ...


def train(env, config, seed, policy=None):  # called once per stage
    set_seed(seed)  # makes your run reproducible
    if policy is None:  # first stage: you create the policy
        policy = SharedPolicy(...)
    ...  # continue training it on this stage's households
    return policy, {}  # the trained policy + a dict of saved paths
```

### Rules
- Import only from the Python standard library, the packages in [pyproject.toml](pyproject.toml) and `course_package`, in the versions `uv sync` installs. Do not import from other folders or files you create and do not rely on notebook variables: the grader runs your file on its own.
- Keep all your code on a **single self-contained python file**. This is what you will submit for grading.
- Save anything you want to keep (checkpoints, plots) under `outputs/`. To load a PyTorch checkpoint that your own run saved, use `torch.load(path, map_location="cpu", weights_only=False)`.

## 4. How you are graded
1. **Independent training runs.** Your algorithm is run 3 times, using different random seeds. Each run starts from scratch in its own isolated process. Every run follows the complete curriculum, consisting of 5 stages, with one additional training household introduced at each stage.

    > **Training budget.** 20k steps (interactions with the RL environment) or 1 min per stage. As soon as any of the stopping criteria are met, your algorithm will be stopped and graded on the policy it found up to then.  Total budget per seed: 100k steps / 5 min.

2. **Performance metric.** We evaluate each episode using the \textbf{total undiscounted return} summed across all agents. The final test score is the \textbf{mean return} over all test episodes. Higher returns indicate better performance.
3. **Generalisation.** The final policy obtained at the end of each run is evaluated on two groups of households:
    * Training households (5): the households encountered during training with unseen episodes. This measures how in-distribution generalization $(S^{train}_k)$.
    * Unseen households (5): households excluded from training. This measures out-of-distribution generalization $(S^{new}_k)$.
    
    The two evaluation groups contribute to the run's score $(S_k)$ according to a predefined weighted sum.
$$
S_k = w^{train}\,S^{train}_k+w^{new}\,S^{new}_k
$$

4. **Final group score.** Your group's final score is calculated by averaging the scores from the 3 independent runs. Groups are ranked according to this final score, with higher scores being better.

    > **Failed runs.** A run is considered failed if it crashes and the file cannot be successfully executed. Failed runs are excluded from the final grade average but imply an additional penalty in the final group score. Therefore, ensure that your implementation is robust across all random seeds.

## 5. If something goes wrong

- **The first `uv sync` is slow.** It downloads Python 3.12, CPU-only PyTorch and the coursework dependencies. Later runs reuse uv's cache.
- **Port 8888 is busy.** Jupyter picks the next free port automatically; use the link it prints, or choose one with `uv run jupyter lab --port 8899`.
- **`export_submission` complains.** Your file must define both `AlgorithmConfig` and `train(env, config, seed, policy=None)`. Check syntax.
- **My file works in the notebook but fails when graded.** It probably depends on something outside the file: an import from `algorithms/` or another file, a notebook variable, or a read from `data/`. Move that code into the file.
- **The repository was updated.** Save your algorithm, run `git pull`, and then run `uv sync`.
- **My variables are gone after a break.** Expected — see the note in [Section 1](#1-start-the-coursework). 
