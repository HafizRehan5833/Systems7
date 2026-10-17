# System7 Internship

This repository contains learning exercises, experiments, and deliverables
completed during the System7 internship.

## Repository structure

```text
systems7/
└── task1/
    ├── day1/        Python basics
    ├── day2/        Calculator game
    ├── day3/        Data structures
    ├── day4/        Classes and student management
    ├── day5/        NumPy operations
    ├── day6/        Data cleaning
    └── day7/        Data analysis and visualisation
```

## Current work

### Task 1

Task 1 builds Python and data-analysis foundations. It includes:

- Python syntax, variables, operators, input, and control flow
- Lists, dictionaries, tuples, and sets
- Object-oriented programming with small management applications
- NumPy arrays and numerical operations
- Pandas data cleaning and CSV processing
- Matplotlib and Seaborn charts
- Summary statistics and correlation analysis
- An optional introductory TensorFlow script

See the detailed instructions and commands in
[`task1/README.md`](./task1/README.md).

## Development setup

The project uses `uv` for Python environment and dependency management.
From the task directory:

```powershell
cd task1
uv sync
```

Run an exercise with:

```powershell
uv run python .\day1\basics.py
```

## Internship workflow

1. Create or update work in the appropriate task and day directory.
2. Keep scripts small, readable, and runnable from the documented command.
3. Record generated reports and visualisations with the related exercise.
4. Verify changes by running the affected script or test.
5. Update the relevant README when setup or usage changes.

## Notes

- Generated datasets and charts are kept next to the script that produces them.
- TensorFlow is optional for the current task and is not part of the default
  dependency set.
- Do not commit credentials, private keys, or other sensitive information.
