import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import tkinter as tk
from tkinter import ttk, messagebox

x = sp.symbols('x')

def get_function(expr_str):
    expr = sp.sympify(expr_str)
    return sp.lambdify(x, expr, 'numpy'), expr

def show_plot_with_table(f, expr_str, x_vals, highlight=None, title="Method Plot", table_data=None):
    # Create the main window
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1000x700")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))
    y_vals = f(x_vals)

    # Plot the function
    ax.plot(x_vals, y_vals, label=f'f(x) = {expr_str}')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()

    # Highlight points
    if highlight:
        for point in highlight:
            ax.axvline(x=point, linestyle='--', color='red', alpha=0.7)
            ax.plot(point, f(point), 'ro')
            ax.text(point, f(point), f"x ≈ {point:.4f}", fontsize=9, ha='right', va='bottom', color='blue')

    # Embed the matplotlib plot into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Table area with scrollbar
    if table_data:
        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)

        columns = table_data[0]
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=8)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)

        for row in table_data[1:]:
            tree.insert('', tk.END, values=row)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        tree.pack(side='left', fill='both', expand=True)

    win.mainloop()

# Numerical Methods
def graphical_method(expr_str):
    f, _ = get_function(expr_str)
    x_vals = np.linspace(-10, 10, 1000)
    y_vals = f(x_vals)

    roots = []
    for i in range(1, len(x_vals)):
        if y_vals[i-1] * y_vals[i] < 0:
            root_est = (x_vals[i-1] + x_vals[i]) / 2
            roots.append(root_est)

    show_plot_with_table(f, expr_str, x_vals, highlight=roots, title="Graphical Method")

def incremental_method(expr_str, x0, dx, max_iter=100):
    f, _ = get_function(expr_str)
    roots = []
    iteration_data = []

    for i in range(max_iter):
        x1 = x0 + dx
        fx0 = f(x0)
        fx1 = f(x1)
        iteration_data.append([i+1, f"{x0:.6f}", f"{x1:.6f}", f"{fx0:.6f}", f"{fx1:.6f}"])
        if fx0 * fx1 < 0:
            roots = [(x0 + x1) / 2]
            break
        x0 = x1

    x_vals = np.linspace(x0 - 10*dx, x0 + 10*dx, 500)
    table = [["Iteration", "x0", "x1", "f(x0)", "f(x1)"]] + iteration_data
    show_plot_with_table(f, expr_str, x_vals, highlight=roots, title="Incremental Method", table_data=table)

def bisection_method(expr_str, a, b, tol=1e-6, max_iter=100):
    f, _ = get_function(expr_str)
    if f(a) * f(b) > 0:
        messagebox.showerror("Error", "Invalid interval. f(a)*f(b) > 0")
        return

    iteration_data = []
    history = []

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        history.append(c)
        iteration_data.append([i+1, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6f}"])
        if abs(fc) < tol:
            break
        if f(a) * fc < 0:
            b = c
        else:
            a = c

    x_vals = np.linspace(a - 1, b + 1, 500)
    table = [["Iteration", "a", "b", "c", "f(c)"]] + iteration_data
    show_plot_with_table(f, expr_str, x_vals, highlight=[history[-1]], title="Bisection Method", table_data=table)

def regula_falsi(expr_str, a, b, tol=1e-6, max_iter=100):
    f, _ = get_function(expr_str)
    if f(a) * f(b) > 0:
        messagebox.showerror("Error", "Invalid interval. f(a)*f(b) > 0")
        return

    iteration_data = []
    history = []

    for i in range(max_iter):
        fa, fb = f(a), f(b)
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        history.append(c)
        iteration_data.append([i+1, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6f}"])
        if abs(fc) < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c

    x_vals = np.linspace(a - 1, b + 1, 500)
    table = [["Iteration", "a", "b", "c", "f(c)"]] + iteration_data
    show_plot_with_table(f, expr_str, x_vals, highlight=[history[-1]], title="Regula Falsi Method", table_data=table)

def newton_raphson_with_table(expr_str, x0, tol=1e-6, max_iter=100):
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, 'numpy')
    f_prime = sp.lambdify(x, sp.diff(expr, x), 'numpy')

    iteration_data = []
    history = [x0]

    for i in range(max_iter):
        fx = f(x0)
        fpx = f_prime(x0)
        if fpx == 0:
            messagebox.showerror("Error", "Zero derivative encountered.")
            return
        iteration_data.append([i+1, f"{x0:.6f}", f"{fx:.6f}", f"{fpx:.6f}"])
        x1 = x0 - fx / fpx
        history.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1

    x_vals = np.linspace(min(history)-2, max(history)+2, 500)
    table = [["Iteration", "x", "f(x)", "f'(x)"]] + iteration_data
    show_plot_with_table(f, expr_str, x_vals, highlight=[history[-1]], title="Newton-Raphson Method", table_data=table)

def secant_method_with_table(expr_str, x0, x1, tol=1e-6, max_iter=100):
    f, _ = get_function(expr_str)
    history = [x0, x1]
    iteration_data = []

    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0:
            messagebox.showerror("Error", "Division by zero in Secant Method.")
            return
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)
        iteration_data.append([i+1, f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", f"{fx2:.6f}"])
        history.append(x2)
        if abs(fx2) < tol:
            break
        x0, x1 = x1, x2

    x_vals = np.linspace(min(history)-2, max(history)+2, 500)
    table = [["Iteration", "x0", "x1", "x2", "f(x2)"]] + iteration_data
    show_plot_with_table(f, expr_str, x_vals, highlight=[history[-1]], title="Secant Method", table_data=table)

# GUI Menu
def launch_gui_menu():
    root = tk.Tk()
    root.title("Numerical Methods GUI")

    methods = {
        "Graphical Method": lambda: graphical_method(expr_entry.get()),
        "Incremental Method": lambda: incremental_method(expr_entry.get(), float(param1.get()), float(param2.get())),
        "Bisection Method": lambda: bisection_method(expr_entry.get(), float(param1.get()), float(param2.get())),
        "Regula Falsi Method": lambda: regula_falsi(expr_entry.get(), float(param1.get()), float(param2.get())),
        "Newton-Raphson Method": lambda: newton_raphson_with_table(expr_entry.get(), float(param1.get())),
        "Secant Method": lambda: secant_method_with_table(expr_entry.get(), float(param1.get()), float(param2.get()))
    }

    tk.Label(root, text="Function f(x):").pack()
    expr_entry = tk.Entry(root, width=40)
    expr_entry.pack()

    tk.Label(root, text="Parameter 1 (x0 or a):").pack()
    param1 = tk.Entry(root, width=20)
    param1.pack()

    tk.Label(root, text="Parameter 2 (dx, b or x1):").pack()
    param2 = tk.Entry(root, width=20)
    param2.pack()

    for name, func in methods.items():
        tk.Button(root, text=name, command=func, width=30).pack(pady=2)

    root.mainloop()

# Start GUI
launch_gui_menu()
