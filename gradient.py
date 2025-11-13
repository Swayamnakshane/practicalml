

# Parameters for Gradient Descent
x = 2                # Initial guess
learning_rate = 0.1  # Step size
epochs = 50          # Number of iterations

# For plotting
x_vals = [x]
y_vals = [func(x)]




# Gradient Descent Loop
for i in range(epochs):
    dx = grad(x)
    x = x - learning_rate * dx

    x_vals.append(x)
    y_vals.append(func(x))

    print(f"Iteration {i+1}: x = {x:.5f}, f(x) = {func(x):.5f}")





print(f"\nLocal minimum at x = {x:.5f}, f(x) = {func(x):.5f}")





# Plotting the function and steps
x_plot = [i for i in range(-10, 5)]
y_plot = [func(i) for i in x_plot]

plt.figure(figsize=(8,5))
plt.plot(x_plot, y_plot, label="y = (x + 3)^2")
plt.scatter(x_vals, y_vals, color='red', label="Gradient Descent Steps")
plt.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=0.6)
plt.title("Gradient Descent to Find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig('assign4.png')
plt.show()
