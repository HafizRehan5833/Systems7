# Day 2 – Over/Underfitting, Preprocessing and Tuning (Module A)

**Commit:** `day-16-tuning-model-comparison`

## Topics
- Overfitting, underfitting, and generalization
- Feature scaling and encoding categorical data
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Practical task: compare multiple models and justify the best one

## Practice questions
| # | Level | File | Question | Run Record |
| --- | --- | --- | --- | --- |
| 6 | Medium | `practice/q06_tree_depth_overfitting.py` | Decision tree depth 1 to 15: train vs test score | [`Run_file/q06.png`](Run_file/q06.png) |
| 7 | Medium | `practice/q07_column_transformer.py` | Scale numeric and one-hot encode categorical in one step | [`Run_file/Q07.png`](Run_file/Q07.png) |
| 8 | Medium | `practice/q08_svm_tuning.py` | GridSearchCV vs RandomizedSearchCV for an SVM: score and run time | [`Run_file/Q08.png`](Run_file/Q08.png) |

## Execution Records (`Run_file/`)
All execution records and terminal output screenshots of the practice scripts are maintained in the `Run_file/` folder:
- **Q06 (Tree Depth Overfitting):** `Run_file/q06.png`
- **Q07 (Column Transformer):** `Run_file/Q07.png`
- **Q08 (SVM Tuning):** `Run_file/Q08.png`

## Module A practical task
Folder: `tasks/moduleA_model_comparison/` – compare several models on classification and regression, then justify the best one in `JUSTIFICATION.md`.

## Learning notes
- **Bias–variance trade-off:** Why do models underfit or overfit, and how does model complexity change both? Use the Q6 plot in your note.
  - My note:

## Completed today
- [x] Solved Q6 to Q8 (execution records added to `Run_file/`)
- [ ] Completed the Module A model comparison and wrote the justification
- [ ] Wrote the bias–variance note
