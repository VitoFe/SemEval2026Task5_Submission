# SemEval2026Task5 Submission
> This meta-repository orchestrates the distinct modeling strategies developed for SemEval 2026 Task 5. To ensure **strict experimental isolation** and **reproducibility**, each approach is encapsulated as an independent submodule within the `approaches` directory.
> This architectural decision decouples dependency management, allowing disparate frameworks (e.g., distinct CUDA versions, conflicting library requirements) to coexist without contamination. It facilitates a modular comparative analysis, where individual data pipelines and architectures are preserved in their optimal, version-controlled states.

### Installation & Execution

**Recursive Cloning**
The repository utilizes `git submodules` to enforce architectural isolation. Do not perform a standard clone. Use the `--recurse-submodules` flag to initialize and fetch all nested repositories and dependencies in a single operation:

```bash
git clone --recurse-submodules https://github.com/VitoFe/SemEval2026Task5_Submission.git

```

*If the repository was previously cloned without this flag, reconstruct the dependency tree via:*

```bash
git submodule update --init --recursive

```

**Approach-Specific Documentation**
Execution logic is decoupled. Navigate to the target directory under `approaches/` and consult the local `README.md`. Each submodule contains self-contained instructions for environment instantiation, data preprocessing, and inference specific to that architecture.

---

[Paper detailing our approach](.\Paper_Submission_Group31_Task5.pdf)
