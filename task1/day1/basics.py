
name = "Student"
age = 21
height = 5.8
is_intern = True
print("Types:", type(name), type(age), type(height), type(is_intern))

a, b = 17, 5
print("Add:", a + b, "| Sub:", a - b, "| Mul:", a * b)
print("Div:", a / b, "| Floor:", a // b, "| Mod:", a % b, "| Pow:", a ** b)
print("Comparison a > b:", a > b, "| Logic:", (a > b) and (b > 10))

try:
    user = input("Enter your name: ") or name
except EOFError:
    user = name
print(f"Hello, {user}! Welcome to System 7.")
