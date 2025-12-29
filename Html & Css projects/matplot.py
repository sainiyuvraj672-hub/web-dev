import matplotlib.pyplot as plt

# Monthly sales data
sales = [12, 15, 14, 16, 18, 20, 24, 22, 25, 27, 30, 28]
months = list(range(1, 13))

# ------------------ Line Chart ------------------
plt.figure(figsize=(10, 5))
plt.plot(months, sales)
plt.xlabel("Month")
plt.ylabel("Sales (₹ lakhs)")
plt.title("Monthly Sales - Line Chart")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ Bar Chart ------------------
plt.figure(figsize=(10, 5))
plt.bar(months, sales)
plt.xlabel("Month")
plt.ylabel("Sales (₹ lakhs)")
plt.title("Monthly Sales - Bar Chart")
plt.grid(True)
plt.tight_layout()
plt.show()
