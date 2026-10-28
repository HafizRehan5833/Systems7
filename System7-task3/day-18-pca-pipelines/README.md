# Day 4 – PCA, Feature Engineering, Imbalanced Data, Leakage, Pipelines (Module B)

**Commit:** `day-18-pca-pipelines`

## Topics
- Principal Component Analysis (PCA) and feature selection
- Imbalanced datasets and data leakage
- Scikit-learn pipelines
- Practical task: customer segmentation (in `tasks/moduleB_customer_segmentation/`)

## Practice questions
| # | Level | File | Question |
| --- | --- | --- | --- |
| 11 | Medium | `practice/q11_imbalanced_class_weight.py` | class_weight on imbalanced data: recall before and after |
| 12 | Hard | `practice/q12_pca_2d.py` | PCA to 2 components: plot clusters, explained variance |
| 13 | Hard | `practice/q13_pipeline_leakage.py` | Pipeline vs scaling before the split: show leakage |

## Learning notes
- **Data leakage:** How does information from the test set sneak into training? Why do pipelines prevent it?
- **Curse of dimensionality:** Why do PCA and feature selection help when there are many features?
  - My note:

## Completed today
- [ ] Solved Q11 to Q13
- [ ] Completed the segmentation task and wrote the summary in `tasks/moduleB_customer_segmentation/SEGMENTS.md`
- [ ] Wrote the two notes above
