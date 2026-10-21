"""Day 4 (Task 4): Bank Account class (basic OOP)."""


class InsufficientFunds(Exception):
    pass


class BankAccount:
    bank_name = "System 7 Bank"

    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance
        self.history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self.history.append(("deposit", amount))

    def withdraw(self, amount):
        if amount > self._balance:
            raise InsufficientFunds(f"Balance {self._balance}, requested {amount}")
        self._balance -= amount
        self.history.append(("withdraw", amount))

    @property
    def balance(self):
        return self._balance

    def __str__(self):
        return f"{self.owner}'s account at {self.bank_name}: Rs {self._balance}"


class SavingsAccount(BankAccount):
    def __init__(self, owner, balance=0, rate=0.05):
        super().__init__(owner, balance)
        self.rate = rate

    def add_interest(self):
        self.deposit(self._balance * self.rate)


if __name__ == "__main__":
    acc = SavingsAccount("Ali", 1000)
    acc.deposit(500)
    acc.withdraw(200)
    acc.add_interest()
    print(acc)
    print(acc.history)
    try:
        acc.withdraw(10**6)
    except InsufficientFunds as e:
        print("Error:", e)
