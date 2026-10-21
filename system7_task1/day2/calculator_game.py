import random
import sys


def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b):
    if b == 0:
        return "Error: division by zero"
    return a / b


OPS = {"+": add, "-": subtract, "*": multiply, "/": divide}


def calculator():
    print("\n--- Calculator (type q to quit) ---")
    while True:
        op = input("Operator (+ - * /) or q: ").strip()
        if op == "q":
            break
        if op not in OPS:
            print("Invalid operator.")
            continue
        try:
            a = float(input("First number: "))
            b = float(input("Second number: "))
        except ValueError:
            print("Please enter valid numbers.")
            continue
        print("Result:", OPS[op](a, b))


def guessing_game():
    print("\n--- Number guessing game (1-100) ---")
    secret = random.randint(1, 100)
    for attempt in range(1, 8):
        try:
            guess = int(input(f"Attempt {attempt}/7 - your guess: "))
        except ValueError:
            print("Enter a whole number.")
            continue
        if guess == secret:
            print(f"Correct! You won in {attempt} attempts.")
            return
        print("Too low." if guess < secret else "Too high.")
    print(f"Out of attempts. The number was {secret}.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print(add(2, 3), subtract(9, 4), multiply(3, 4), divide(8, 2), divide(1, 0))
        for i in range(1, 6):
            print(i, "is", "even" if i % 2 == 0 else "odd")
    else:
        choice = input("1) Calculator  2) Guessing game: ")
        calculator() if choice == "1" else guessing_game()
