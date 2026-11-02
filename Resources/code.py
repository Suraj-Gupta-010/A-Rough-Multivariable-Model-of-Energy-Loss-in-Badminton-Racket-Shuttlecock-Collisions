import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

b = 22
resolution = 800

def y_boundary(h):
    t = np.where(h >= 0, 2.0, 2.4)

    base = 1 - np.abs(h / 4)**t
    base = np.maximum(base, 0)

    return 3 * base**(1 / t)


def delta(h, k):
    Y = y_boundary(h)
    d_frame = (np.abs(Y-k) + np.abs(Y+k)) / 2
    d_centre = np.sqrt(h**2 + k**2)
    return (d_frame + d_centre) / 2


def L(a, b):
    x = a - 2.3
    q = b - 20

    return (
        65.27
        - 19.94*x
        + 98.60*x**2
        + 194.41*x**3
        + 2.42*q
        - 9.26*x*q
        - 22.16*x**2*q
    )

h = np.linspace(-4, 4, resolution)
k = np.linspace(-3, 3, resolution)

H, K = np.meshgrid(h, k)

inside_right = (
    (H >= 0) &
    (H**2 / 16 + K**2 / 9 <= 1)
)

inside_left = (
    (H < 0) &
    (np.abs(H / 4)**2.4 + np.abs(K / 3)**2.4 <= 1)
)

mask = inside_right | inside_left

D = delta(H, K)
Z = L(D, b)

Z_masked = np.ma.masked_where(~mask, Z)

cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    ["#00FF00", "#F8F800","#ff9100", "#ff0000"]
)

cmap.set_bad((0, 0, 0, 0))

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    Z_masked,
    extent=[-4, 4, -3, 3],
    origin="lower",
    cmap=cmap,
    interpolation="bilinear",
    aspect="equal"
)

boundary_x = np.linspace(-4, 4, 1200)
boundary_y = y_boundary(boundary_x)

ax.plot(boundary_x, boundary_y, color="black", linewidth=1.5)
ax.plot(boundary_x, -boundary_y, color="black", linewidth=1.5)

# Labels
ax.set_xlabel("h")
ax.set_ylabel("k")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Predicted energy loss (%)")

ax.set_xlim(-4.1, 4.1)
ax.set_ylim(-3.1, 3.1)

plt.tight_layout()
plt.show()
