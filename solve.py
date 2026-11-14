import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Input table: first argument, otherwise test-input.csv next to the script
path = sys.argv[1] if len(sys.argv) > 1 else "test-input.csv"
df = pd.read_csv(path)

# Label is the last column (slab_released)
target = "slab_released" if "slab_released" in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Default hyperparameters, only a fixed integer random_state
clf = DecisionTreeClassifier(random_state=0).fit(X, y)
tree = clf.tree_

# Read the questions in the order the tree asks them:
# depth-first, root first, left branch before right (pre-order)
letters = []
stack = [0]
while stack:
    node = stack.pop()
    if tree.feature[node] < 0:  # leaf, no question here
        continue
    letters.append(X.columns[tree.feature[node]][0])
    stack.append(tree.children_right[node])
    stack.append(tree.children_left[node])

word = "".join(letters).upper()
print(f"SIGMOID_{word}")