import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    alpha, beta = np.pi/6, np.pi/3
    z1 = np.exp(1j * alpha)
    z2 = np.exp(1j * beta)
    z_sum = z1 + z2
    
    fig, ax = plt.subplots(figsize=(6,6))
    pts = [0, z1, z_sum, z2]
    x = [p.real for p in pts] + [0]
    y = [p.imag for p in pts] + [0]
    
    ax.plot(x, y, 'b-o')
    ax.plot([0, z_sum.real], [0, z_sum.imag], 'r--', label='OC')
    ax.plot([z1.real, z2.real], [z1.imag, z2.imag], 'g--', label='AB')
    
    ax.set_aspect('equal')
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.set_title("Argand Diagram of Rhombus OACB")
    ax.legend()
    return fig