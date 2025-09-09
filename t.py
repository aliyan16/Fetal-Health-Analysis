import os
import csv

# Adjust to your dataset folder
images_dir = "C:/AllData/Selfskills/FYP/FetalCLIP/Images"
csv_path = "C:\AllData\Selfskills\FYP\FetalCLIP\FETAL_PLANES_DB_data.csv"

# Define classes (these should match your 5 planes)
classes = ["3VV", "4CH", "Abdominal", "Brain", "LVOT"]

rows = []
for fname in os.listdir(images_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
        # Try to guess the label from the filename
        label = None
        for c in classes:
            if c.lower() in fname.lower():
                label = c
                break
        if label is None:
            label = "Unknown"  # fallback if no match found
        rows.append({"img": os.path.join(images_dir, fname), "label": label})

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["img", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created at: {csv_path} with {len(rows)} entries")
