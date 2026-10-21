# System7 Internship - Task 1

This project contains the first set of Python internship exercises, covering
Python fundamentals, data structures, simple classes, NumPy, Pandas, data
cleaning, and data analysis.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Setup

From this directory:

```powershell
uv sync
```

## Exercises

| Day | File | Topic |
| --- | --- | --- |
| 1 | `day1/basics.py` | Python variables, types, operators, input, and output |
| 2 | `day2/calculator_game.py` | Calculator game and control flow |
| 3 | `day3/data_structures.py` | Python data structures |
| 4 | `day4/bank_account.py` | Classes and bank-account operations |
| 4 | `day4/student_manager.py` | Student management |
| 5 | `day5/numpy_ops.py` | NumPy operations |
| 6 | `day6/clean_data.py` | Create and clean a student dataset |
| 7 | `day7/analysis.py` | CSV analysis, statistics, and charts |
| 7 | `day7/tensorflow_intro.py` | TensorFlow tensors and variables |

## Running the exercises

Run a script from the `task1` directory with:

```powershell
uv run python .\day1\basics.py
uv run python .\day6\clean_data.py
uv run python .\day7\analysis.py
```

`day6/clean_data.py` creates `students_messy.csv`, `students_clean.csv`, and
`before_after.png`. The Day 7 analysis reads `day7/students_clean.csv`, then
creates a summary report and five charts.

The TensorFlow example is optional and requires TensorFlow to be installed
separately. It is not included in the current project dependencies.

## Day 7 results

The analysis uses 190 cleaned student records:

- Mean score: 74.12
- Highest department mean: AI, with 75.40
- Study-hours-to-score correlation: 0.82

Generated reports and charts are stored in `day7/`.