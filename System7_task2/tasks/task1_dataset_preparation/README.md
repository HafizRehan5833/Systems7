# Task 1 – Explore and Prepare a Dataset

**Skills used:** Pandas, NumPy, statistics, data splitting

## Goal
Load the house dataset, explore it, clean it, add a useful label, and split it into training, validation and testing sets.

## Steps the script performs
1. Load `../data/houses.csv` and show its shape, types and summary statistics.
2. Count missing values and fill them with the median (median is robust to outliers).
3. Remove duplicate rows.
4. Create a classification label `is_expensive` (1 if price is above the median) for Task 3.
5. Split into 60% train, 20% validation, 20% test (fixed `random_state` for reproducibility).
6. Save the splits to `data/processed/`.

## Run

From the `System7_task2/tasks` directory:

```bash
uv run python .\task1_dataset_preparation\prepare_dataset.py
```

Or, from the `System7_task2` directory:

```bash
uv run --project .\tasks python .\tasks\task1_dataset_preparation\prepare_dataset.py
```

## Deliverables
- Printed exploration report
- Files in `tasks/data/processed/`: `train.csv`, `val.csv`, `test.csv`
