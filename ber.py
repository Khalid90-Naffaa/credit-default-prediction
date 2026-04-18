import zipfile, json, re, warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

ZIP_PATH = "اخبار مزيفة.zip"

ARABIC_STOP_WORDS = {
    "من","إلى","عن","على","في","مع","هذا","هذه","ذلك","تلك","التي","الذي",
    "الذين","ما","لا","لم","لن","إن","أن","كان","كانت","يكون","تكون",
    "هو","هي","هم","هن","أنا","نحن","أنت","أنتم","وقد","قد","ثم","حتى",
    "إذا","لو","لكن","بل","أو","أم","كما","بما","مما","عند","عندما","حين",
    "بين","كل","بعض","غير","سوى","فقط","أيضا","جدا","ولا","وما","فلا",
    "وهو","وهي","وهم","به","بها","له","لها","منه","منها","عنه","عنها",
    "فيه","فيها","قبل","بعد","وكان","وكانت","وقال","وقالت","اي","أي","يا"
}

def clean_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(t for t in text.split() if t not in ARABIC_STOP_WORDS and len(t) > 2)

print("=" * 60)
print("   Arabic Fake News Detection Pipeline  (AFND)")
print("=" * 60)
print("\n[1/5] Loading dataset from ZIP...")

records = []
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    with zf.open("AFND/sources.json") as f:
        source_labels = dict(json.load(f).items())
    for filepath in [n for n in zf.namelist() if n.endswith("scraped_articles.json")]:
        source = filepath.split("/")[2]
        label  = source_labels.get(source, "undecided")
        if label == "undecided":
            continue
        with zf.open(filepath) as f:
            arts = json.load(f)
        items = arts if isinstance(arts, list) else arts.get("articles", [])
        for a in items:
            text = str(a.get("title","")) + " " + str(a.get("text",""))
            records.append({"text": text, "label": 1 if label == "not credible" else 0})

df = pd.DataFrame(records)
print(f"    Total loaded : {len(df):,}")
print(f"    Credible  (0): {(df['label']==0).sum():,}")
print(f"    Fake      (1): {(df['label']==1).sum():,}")

print("\n[2/5] Arabic text pre-processing...")
df["clean_text"] = df["text"].apply(clean_arabic)
df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
print(f"    After cleaning: {len(df):,} articles")

print("\n[3/5] TF-IDF Vectorization + 80/20 Split...")
X_tr, X_te, y_tr, y_te = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True)
Xtr = tfidf.fit_transform(X_tr)
Xte = tfidf.transform(X_te)
print(f"    Train: {Xtr.shape[0]:,} | Test: {Xte.shape[0]:,} | Features: {Xtr.shape[1]:,}")

results = {}

print("\n[4/5] Training Models...")
print("\n━━ Naive Bayes (Baseline) ━━")
nb = MultinomialNB()
nb.fit(Xtr, y_tr)
nb_pred = nb.predict(Xte)
results["Naive Bayes"] = {"Accuracy": accuracy_score(y_te, nb_pred),
                           "F1-Score": f1_score(y_te, nb_pred, average='weighted')}
print(classification_report(y_te, nb_pred, target_names=["Credible","Fake"]))

print("\n━━ SVM — Linear Kernel ━━")
svm = LinearSVC(max_iter=2000, random_state=42)
svm.fit(Xtr, y_tr)
svm_pred = svm.predict(Xte)
results["SVM"] = {"Accuracy": accuracy_score(y_te, svm_pred),
                   "F1-Score": f1_score(y_te, svm_pred, average='weighted')}
print(f"    Accuracy : {results['SVM']['Accuracy']:.4f}")
print(f"    F1-Score : {results['SVM']['F1-Score']:.4f}")

print("\n━━ Random Forest (100 estimators) ━━")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(Xtr, y_tr)
rf_pred = rf.predict(Xte)
results["Random Forest"] = {"Accuracy": accuracy_score(y_te, rf_pred),
                              "F1-Score": f1_score(y_te, rf_pred, average='weighted')}
print(f"    Accuracy : {results['Random Forest']['Accuracy']:.4f}")
print(f"    F1-Score : {results['Random Forest']['F1-Score']:.4f}")

print("\n[5/5] Comparison Summary")
print("=" * 50)
summary = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
print(summary.to_string(float_format="{:.4f}".format))
best = max(results, key=lambda k: results[k]["Accuracy"])
print(f"\n Best Model : {best}")
print(f"  Accuracy  : {results[best]['Accuracy']:.4f}")
print(f"  F1-Score  : {results[best]['F1-Score']:.4f}")
print("=" * 50)

print("\nGenerating bar chart...")
models = list(results.keys())
accs   = [results[m]["Accuracy"] for m in models]
f1s    = [results[m]["F1-Score"]  for m in models]
colors = ["#4C9BE8", "#E85C4C", "#4CE87A"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Arabic Fake News Detection — Model Comparison", fontsize=14, fontweight="bold")

for ax, vals, title, ylabel in zip(
    axes, [accs, f1s],
    ["Accuracy Comparison", "F1-Score Comparison"],
    ["Accuracy", "F1-Score (Weighted)"]
):
    bars = ax.bar(models, vals, color=colors, edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to: model_comparison.png")
print("\nDone!")
