"""Day 4 (Task 3): student record manager that saves to a file (JSON)."""
import json
import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.json")


def load():
    try:
        with open(FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("Warning: file corrupted, starting fresh.")
        return {}


def save(records):
    with open(FILE, "w") as f:
        json.dump(records, f, indent=2)


def add_student(records, sid, name, grade):
    if sid in records:
        raise ValueError(f"ID {sid} already exists")
    records[sid] = {"name": name, "grade": float(grade)}
    save(records)


def delete_student(records, sid):
    del records[sid]  # KeyError if missing
    save(records)


if __name__ == "__main__":
    recs = load()
    for sid, name, grade in [("1", "Ali", 85), ("2", "Sara", 92)]:
        try:
            add_student(recs, sid, name, grade)
        except ValueError as e:
            print("Skipped:", e)
    try:
        delete_student(recs, "99")
    except KeyError:
        print("Cannot delete: student 99 not found")
    print("Saved records:", load())
