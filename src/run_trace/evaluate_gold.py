import json
import os
import ast
from cycler import cycler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils import combine_spans


# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

plt.rcParams['axes.linewidth'] = 1.5
palette = [
    "#C83A3A",
    "#3F6FA6",
    "#8A8A8A",
    '#7FA68B',
    "#F4C7C3",

   
    "#A62C2C",
    "#2E2E2E",
    "#F4C7C3",
]
plt.rcParams["axes.prop_cycle"] = cycler(color=palette)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def span_length_sum(spans):
    return sum(end - start for start, end in spans)


def get_metrics(df_, column):
    df = df_.copy()
    df["total"] = df[column].apply(span_length_sum)
    df = df[df["total"] != 0]

    lengths = [
        end - start
        for spans in df[column]
        for start, end in spans
    ]

    print("=" * 60)
    print(column)
    print("=" * 60)

    print(df["total"].describe())
    print("total:", df["total"].sum(), df['full_note_text'].apply(len).sum())
    print("Mean span length:", np.mean(lengths) if lengths else 0)

    return df


def load_labelstudio_annotations(json_files, remove_threshold=0):
    annotations = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        for item in data:
            spans = [
                (r["value"]["start"], r["value"]["end"])
                for r in item["annotations"][0]["result"]
                if r["value"]["labels"][0] == "Templated"
            ]

            spans = combine_spans(
                spans,
                item["data"]["text"],
                REMOVE_THRESHOLD=remove_threshold
            )

            annotations.append({
                "note_csn_id": int(item["data"]["note_csn_id"]),
                "full_note_text": item["data"]["text"],
                "spans": spans
            })

    return pd.DataFrame(annotations)


def get_intervals(x):
    return [(item["start"], item["end"]) for item in x]


def intervals_to_set(intervals):
    active = set()
    for start, end in intervals:
        active.update(range(start, end))
    return active



def get_performance(reference_intervals, prediction_intervals):
    """
    Calculates precision and recall with different span filters:
      - Precision: computed only over all intervals
      - Recall:    computed over intervals longer than 50

    Returns: precision, recall
    """
    long_reference_intervals = [(s, e) for s, e in reference_intervals if (e - s) > 50]

    # recall > 50
    long_ref_points = {p for s, e in long_reference_intervals for p in (s, e)}
    long_pred_points = {p for s, e in prediction_intervals for p in (s, e)}
    long_points = long_ref_points | long_pred_points


    if long_points:
        long_ref_set = intervals_to_set(long_reference_intervals)
        pred_set = intervals_to_set(prediction_intervals)

        tp_recall = sum(1 for t in range(min(long_points), max(long_points)) if t in long_ref_set and t in pred_set)
        fn_recall = sum(1 for t in range(min(long_points), max(long_points)) if t in long_ref_set and t not in pred_set)


    all_ref_points = {p for s, e in reference_intervals for p in (s, e)}
    all_pred_points = {p for s, e in prediction_intervals for p in (s, e)}
    all_points = all_ref_points | all_pred_points


    if all_points:
        ref_set = intervals_to_set(reference_intervals)
        pred_set = intervals_to_set(prediction_intervals)

        tp_prec = sum(1 for t in range(min(all_points), max(all_points)) if t in ref_set and t in pred_set)
        fp_prec = sum(1 for t in range(min(all_points), max(all_points)) if t not in ref_set and t in pred_set)


    return tp_recall, fn_recall, tp_prec, fp_prec

def plot(predictions):
    melted = predictions.melt(
        id_vars='note_csn_id',
        var_name='metric',
        value_name='value',
        value_vars=[
            # 'stage1_recall', 'stage1_precision', 'stage1_f1',
            # 'stage2_recall', 'stage2_precision', 'stage2_f1',
            # 'trace_recall', 'trace_precision', 'trace_f1',
             'stage1_recall', 'stage1_precision',
            'stage2_recall', 'stage2_precision',
            'trace_recall', 'trace_precision',
        ]
    )

    melted[['arm', 'metric']] = melted['metric'].str.split('_', expand=True)
    print(melted.head())
    predictions[['note_csn_id','stage1_recall', 'stage1_precision', 
                #  'stage1_f1',
            'stage2_recall', 'stage2_precision', 
            # 'stage2_f1',
            'trace_recall', 'trace_precision'
            # 'trace_f1'
            ]].to_csv('../analysis/gold_note_metics.csv', index=False)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.boxplot(melted, x='metric', y='value', hue='arm',ax=ax)
    plt.legend(loc='upper center', bbox_to_anchor=(.5,1.2), ncol=3)
    plt.savefig('../analysis/gold_hist.png', dpi=300, bbox_inches='tight')
    return


def bootstrap_ci(tp_series, other_series, n_boot=1000, ci=95, random_state=42):

    rng = np.random.default_rng(random_state)

    mask = tp_series.notna() & other_series.notna()
    tp_vals = tp_series[mask].values
    other_vals = other_series[mask].values
    n = len(tp_vals)

    boot_metrics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tp = tp_vals[idx].sum()
        other = other_vals[idx].sum()
        denom = tp + other
        boot_metrics.append(tp / denom if denom > 0 else 0.0)

    lower = np.percentile(boot_metrics, (100 - ci) / 2)
    upper = np.percentile(boot_metrics, 100 - (100 - ci) / 2)

    return lower, upper

def calculate_recall(tp, fn):
    return tp / (tp+fn)

def calculate_precision(tp, fp):
    return tp / (tp+fp)

def main():

    BASE_PATH = "/Users/cahoon/Documents/research/phi/templating/label_studio"

    print("---------------------------")
    print("Load Chloe annotations")
    print("---------------------------")
    to_keep = pd.read_csv(f"{BASE_PATH}/chloe_keep.csv")

    chloe_jsons = [
        f"{BASE_PATH}/chloe1.json",
        f"{BASE_PATH}/chloe2.json",
    ]

    chloe_annotations = load_labelstudio_annotations(chloe_jsons)
    print("To keep: ", to_keep.shape[0])
    print("Total Annotations: ", chloe_annotations.shape[0])
    chloe_annotations = chloe_annotations[
        chloe_annotations["note_csn_id"].isin(to_keep["note_csn_id"])
    ].drop_duplicates("note_csn_id", keep="last")
    print("After filtering: ", chloe_annotations.shape[0])


    print("---------------------------")
    print("Load Jordan annotations")
    print("---------------------------")
    to_remove = pd.read_csv(f"{BASE_PATH}/jordan_skip.csv")

    jordan_jsons = [
        f"{BASE_PATH}/jordan.json",
    ]

    jordan_annotations = load_labelstudio_annotations(jordan_jsons)
    jordan_annotations = jordan_annotations.drop_duplicates("note_csn_id", keep="last")
    print("To remove: ", to_remove.shape[0])
    print("Total Annotations: ", jordan_annotations.shape[0])
    jordan_annotations = jordan_annotations[
        ~jordan_annotations["note_csn_id"].isin(to_remove["note_csn_id"])
    ]
    print("After removal: ", jordan_annotations.shape[0])


    print("---------------------------")
    print("Load Asad annotations")
    print("---------------------------")
    to_remove = pd.read_csv(f"{BASE_PATH}/asad_skip.csv")

    asad_jsons = [
        f"{BASE_PATH}/asad.json",
    ]

    asad_annotations = load_labelstudio_annotations(asad_jsons)
    asad_annotations = asad_annotations.drop_duplicates("note_csn_id", keep="last")
    print("To remove: ", to_remove.shape[0])
    print("Total Annotations: ", asad_annotations.shape[0])
    asad_annotations = asad_annotations[
        ~asad_annotations["note_csn_id"].isin(to_remove["note_csn_id"])
    ]

    print("After removal: ", asad_annotations.shape[0])
    # Combine
    annotations = pd.concat(
        [chloe_annotations, jordan_annotations, asad_annotations],
        ignore_index=True
    ).drop_duplicates("note_csn_id", keep="last")

    annotations = get_metrics(annotations, "spans")

    annotations.to_csv(f"{BASE_PATH}/combined.csv", index=False)
    print("---------------------------")
    print("Total annotations: ", annotations.shape[0])
    print("---------------------------")

    # ---------------------------
    # Load predictions
    # ---------------------------
    FILTERED = f"{BASE_PATH}/filtered.csv"
    STAGE1_JSONL = f"{BASE_PATH}/gold_sample_stage1.jsonl"
    STAGE2_JSONL = f"{BASE_PATH}/gold_sample_stage2.jsonl"

    note_ids = set(annotations["note_csn_id"])
    filtered_list = {}

    if not os.path.exists(FILTERED):

        with open(STAGE1_JSONL, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["note_csn_id"] in note_ids:
                    filtered_list[item["note_csn_id"]] = {
                        "note_csn_id": item["note_csn_id"],
                        "template_stage1_spans": item["template_spans"],
                        "template_string": item["template_string"],
                        "copyforward_stage1_spans": item["copyforward_spans"],
                    }

        with open(STAGE2_JSONL, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["note_csn_id"] in note_ids:
                    filtered_list[item["note_csn_id"]].update({
                        "template_stage2_spans": item["template_spans_stage2"],
                        "copyforward_stage2_spans": item["copyforward_spans_stage2"],
                    })

        pd.DataFrame(filtered_list.values()).to_csv(FILTERED, index=False)

    predictions = pd.read_csv(FILTERED)

    span_columns = [
        "template_stage1_spans",
        "copyforward_stage1_spans",
        "template_stage2_spans",
        "copyforward_stage2_spans",
    ]

    for col in span_columns:
        predictions[col] = predictions[col].apply(ast.literal_eval)
        predictions[col] = predictions[col].apply(get_intervals)

    predictions = annotations.merge(predictions, how="inner")
    get_metrics(predictions, "spans")
    # ---------------------------
    # Build prediction arms
    # ---------------------------
    for arm in ["stage1", "trace"]:

        predictions[arm] = [[] for _ in range(len(predictions))]

        if arm in ["stage1", "trace"]:
            predictions[arm] += predictions["template_stage1_spans"]

        if arm in ["stage2", "trace"]:
            predictions[arm] += predictions["template_stage2_spans"]

        predictions[arm] = predictions.apply(
            lambda x: combine_spans(x[arm], x["full_note_text"],  REMOVE_THRESHOLD=50),
            axis=1
        )\

        get_metrics(predictions, arm)

    predictions.to_csv(f'{BASE_PATH}/predictions.csv', index=False)

    # ---------------------------   
    # Evaluation
    # ---------------------------
    precision_results = []
    recall_results = []
   
    for arm in ["stage1", "trace"]:

        predictions[
            [f"{arm}_tp_recall", f"{arm}_fn_recall", f"{arm}_tp_prec", f"{arm}_fp_prec"]
        ] = predictions.apply(
            lambda x: get_performance(x["spans"], x[arm]),
            axis=1,
            result_type="expand"
        )

        # calculate precision and recall
        precision = calculate_precision(predictions[f"{arm}_tp_prec"].sum(), predictions[f"{arm}_fp_prec"].sum())
        recall = calculate_precision(predictions[f"{arm}_tp_recall"].sum(), predictions[f"{arm}_fn_recall"].sum())


        precision_ci = bootstrap_ci(predictions[f"{arm}_tp_prec"], predictions[f"{arm}_fp_prec"])
        recall_ci = bootstrap_ci(predictions[f"{arm}_tp_recall"],  predictions[f"{arm}_fn_recall"])
        # f1_ci = bootstrap_ci(predictions[f"{arm}_f1"])
   
        print("=" * 30)
        print(arm)
        print("=" * 30)
        print(f"precision: {precision:.2f} (95% CI: {precision_ci[0]:.2f}–{precision_ci[1]:.2f})")
        print(f"recall:    {recall:.2f} (95% CI: {recall_ci[0]:.2f}–{recall_ci[1]:.2f})")

        precision_results.append({
            'arm': arm,
            'mean': precision,
            'ci_lower': precision_ci[0],
            'ci_upper': precision_ci[1],
        })

        recall_results.append({
            'arm': arm,
            'mean': recall,
            'ci_lower': recall_ci[0],
            'ci_upper': recall_ci[1],
        })
    # plot(predictions)

    rename_map = {
        'trace': 'TRACE',
        'stage1': 'Reference Module',
        'stage2': 'Frequency Module'
    }

    pd.DataFrame(precision_results) \
        .round(2) \
        .replace({'arm': rename_map}) \
        .to_csv('../analysis/gold_bootstrap_p.csv', index=False)

    pd.DataFrame(recall_results) \
        .round(2) \
        .replace({'arm': rename_map}) \
        .to_csv('../analysis/gold_bootstrap_r.csv', index=False)


if __name__ == "__main__":
    main()


