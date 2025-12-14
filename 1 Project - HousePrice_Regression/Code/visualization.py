"""
visualization.py
Contains simple helper functions for plotting (optional).
"""
import matplotlib.pyplot as plt

def save_scatter_actual_pred(y_true, y_pred, path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, '--')
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted")
    plt.savefig(path); plt.close()
