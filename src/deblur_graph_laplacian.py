import os
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import lil_matrix

np.random.seed(0)


def downsample_image(image, factor):
    """Reduce the image size by a specified factor."""
    width, height = image.size
    new_size = (width // factor, height // factor)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


# Load and downsample the image
img_path = "data/image-Dante.png"
image = Image.open(img_path).convert("L")

factor = 2
image = downsample_image(image, factor)

# Normalize image intensity to [0, 1]
x_true = np.array(image, dtype=float) / 255.0


def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel /= np.sum(kernel)
    return kernel


# Apply Gaussian blur
size = 5
sigma = 1
kernel = gaussian_kernel(size, sigma)

b = convolve2d(
    x_true,
    kernel,
    mode="same",
    boundary="symm"
)

# Add Gaussian noise
noise_level = 0.005
b_delta = b + noise_level * np.random.randn(*b.shape)
b_delta = np.clip(b_delta, 0, 1)

# Right-hand side: A^T b_delta
b_prime = convolve2d(
    b_delta,
    kernel.T,
    mode="same",
    boundary="symm"
)


def weighted_laplacian(image, theta=0.01):
    """Construct a sparse weighted graph Laplacian using 4-neighbor pixels."""
    n, m = image.shape
    L = lil_matrix((n * m, n * m))

    for i in range(n):
        for j in range(m):
            index = i * m + j

            if i > 0:  # pixel above
                weight = np.exp(-((image[i, j] - image[i - 1, j]) ** 2) / theta)
                L[index, index - m] = -weight
                L[index, index] += weight

            if i < n - 1:  # pixel below
                weight = np.exp(-((image[i, j] - image[i + 1, j]) ** 2) / theta)
                L[index, index + m] = -weight
                L[index, index] += weight

            if j > 0:  # pixel left
                weight = np.exp(-((image[i, j] - image[i, j - 1]) ** 2) / theta)
                L[index, index - 1] = -weight
                L[index, index] += weight

            if j < m - 1:  # pixel right
                weight = np.exp(-((image[i, j] - image[i, j + 1]) ** 2) / theta)
                L[index, index + 1] = -weight
                L[index, index] += weight

    return L.tocsr()


# Build graph Laplacian from observed blurred/noisy image
n, m = x_true.shape
#L = weighted_laplacian(b_delta)

y_for_graph = convolve2d(
    b_delta,
    gaussian_kernel(3, 0.8),
    mode="same",
    boundary="symm",
)

L = weighted_laplacian(y_for_graph)

mu = 0.1


def apply_system_matrix(v):
    """Apply A'v = A^T A v + mu L v."""
    x = v.reshape((n, m))

    Ax = convolve2d(
        x,
        kernel,
        mode="same",
        boundary="symm",
    )

    ATAx = convolve2d(
        Ax,
        kernel.T,
        mode="same",
        boundary="symm",
    )

    Lx = L @ v

    return ATAx.ravel() + mu * Lx


A_prime = LinearOperator(
    shape=(n * m, n * m),
    matvec=apply_system_matrix,
    dtype=np.float64,
)

# Solve using conjugate gradient
x_estimated, info = cg(
    A_prime,
    b_prime.ravel(),
    maxiter=1000,
)

x_estimated = x_estimated.reshape(n, m)
x_estimated = np.clip(x_estimated, 0, 1)

def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(1.0 / mse)

print("Blurred/noisy PSNR:", psnr(x_true, b_delta))
print("Restored PSNR:", psnr(x_true, x_estimated))


if info == 0:
    print("CG solver converged.")
else:
    print(f"CG solver did not fully converge. info = {info}")


# Show results
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(x_true, cmap="gray", vmin=0, vmax=1)
plt.title("Original Down-Sampled Image")
plt.axis("off")

plt.subplot(132)
plt.imshow(b_delta, cmap="gray", vmin=0, vmax=1)
plt.title("Blurred / Noisy Image")
plt.axis("off")

plt.subplot(133)
plt.imshow(x_estimated, cmap="gray", vmin=0, vmax=1)
plt.title("Restored Image")
plt.axis("off")

plt.tight_layout()
plt.show()

os.makedirs("results", exist_ok=True)

plt.imsave("results/original.png", x_true, cmap="gray", vmin=0, vmax=1)
plt.imsave("results/blurred_noisy.png", b_delta, cmap="gray", vmin=0, vmax=1)
plt.imsave("results/restored.png", x_estimated, cmap="gray", vmin=0, vmax=1)

with open("results/metrics.txt", "w") as f:
    f.write(f"Blurred/noisy PSNR: {psnr(x_true, b_delta):.4f}\n")
    f.write(f"Restored PSNR: {psnr(x_true, x_estimated):.4f}\n")
    f.write(f"CG info: {info}\n")