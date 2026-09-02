"""
Generates a fixed, deterministic dataset for the AI DevOps lab.
Run once during environment setup (baked into the image at docker build
time by the Dockerfile). Every learner/VM gets identical data, which is
what makes the golden-path accuracy numbers in the prerequisites guide
reproducible.
"""
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=4000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.55, 0.45],
    flip_y=0.03,
    random_state=42,
)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y

# Must match the path baked into the image (Dockerfile WORKDIR /opt/lab)
# and the default LAB_DATASET_PATH read by scripts/train.py.
OUT_PATH = "/opt/lab/dataset/churn_dataset.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Dataset written to {OUT_PATH}:", df.shape)
