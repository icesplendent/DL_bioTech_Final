import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import coherent, basis, Qobj, wigner

def generate_cat_state(alpha, cutoff=50):
    psi_alpha = coherent(cutoff, alpha)
    psi_neg_alpha = coherent(cutoff, -alpha)
    
    cat_state = (psi_alpha + psi_neg_alpha).unit()
    return cat_state

# Parameters
alphas = [2, 3, 4]
cutoff = 50  # cutoff dimension
xvec = np.linspace(-5, 5, 200)

# 2D & 3D
fig, axes = plt.subplots(2, len(alphas), figsize=(15, 10), subplot_kw={'projection': None})
for i, alpha in enumerate(alphas):
    cat_state = generate_cat_state(alpha, cutoff)
    
    W = wigner(cat_state, xvec, xvec)
    X, P = np.meshgrid(xvec, xvec)
    
    # 2D
    ax2d = fig.add_subplot(2, len(alphas), i + 1)
    cont = ax2d.contourf(xvec, xvec, W, levels=100, cmap='RdBu_r')
    ax2d.set_title(f"Cat State (α={alpha}) - 2D")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("p")
    plt.colorbar(cont, ax=ax2d)
    
    # 3D
    ax3d = fig.add_subplot(2, len(alphas), len(alphas) + i + 1, projection='3d')
    ax3d.plot_surface(X, P, W, cmap='RdBu_r', edgecolor='k', alpha=0.8)
    ax3d.set_title(f"Cat State (α={alpha}) - 3D")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("p")
    ax3d.set_zlabel("W(x, p)")
    ax3d.view_init(30, 210)

plt.tight_layout()
plt.show()
