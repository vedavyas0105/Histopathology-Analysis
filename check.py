import numpy as np
import matplotlib.pyplot as plt

images_path = r'data\raw\folds\Fold 1\images\fold1\images.npy'
masks_path  = r'data\raw\folds\Fold 1\masks\fold1\masks.npy'

# ==== 🔹 Step 2: Load the .npy files ====
images = np.load(images_path, allow_pickle=True)
masks = np.load(masks_path, allow_pickle=True)

print("✅ Data Loaded Successfully!")
print(f"Images Shape: {images.shape}")
print(f"Masks Shape:  {masks.shape}")

# ==== 🔹 Step 3: View a few samples ====
num_samples = 3
plt.figure(figsize=(10, 4))

for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(images[i].astype(np.uint8))
    plt.title(f"Image {i}")
    plt.axis("off")

    # Combine all 6 mask channels into one binary mask
    binary_mask = (np.max(masks[i], axis=2) > 0).astype(np.uint8)
    plt.subplot(2, num_samples, num_samples + i + 1)
    plt.imshow(binary_mask, cmap="gray")
    plt.title(f"Mask {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()
