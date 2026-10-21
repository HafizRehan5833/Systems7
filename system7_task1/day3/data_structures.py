"""Day 3: lists, tuples, dictionaries, sets."""

# Lists (mutable, ordered)
marks = [78, 85, 92, 66]
marks.append(88); marks.sort()
print("List:", marks, "| max:", max(marks), "| avg:", sum(marks) / len(marks))
print("Slice:", marks[1:4], "| Reversed:", marks[::-1])

# Tuples (immutable)
point = (10, 20)
x, y = point
print("Tuple unpack:", x, y)

# Dictionaries
student = {"name": "Ali", "age": 20, "courses": ["Python", "Math"]}
student["city"] = "Faisalabad"
for key, value in student.items():
    print(f"  {key}: {value}")

# Sets
s1, s2 = {1, 2, 3, 4}, {3, 4, 5, 6}
print("Union:", s1 | s2, "| Intersection:", s1 & s2, "| Difference:", s1 - s2)
print("Unique letters:", set("mississippi"))

# List comprehension + dict comprehension
squares = [n * n for n in range(1, 6)]
lookup = {n: n ** 2 for n in range(1, 6)}
print(squares, lookup)

# Mini student record manager (in memory; saved to file on Day 4)
records = {}
def add_student(sid, name, grade): records[sid] = {"name": name, "grade": grade}
add_student(1, "Ali", 85); add_student(2, "Sara", 92)
print("Records:", records)
