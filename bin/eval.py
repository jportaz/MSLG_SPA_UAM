import argparse

import pandas as pd
import sacrebleu
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def compute_metrics(path, sep=None, sas_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """
    Reads a file with columns:
      source | target | prediction

    Computes:
      BLEU: hypothesis = prediction, reference = target
      SAS: semantic similarity between prediction and target
    """

    if sep is None:
        sep = "\t" if path.endswith(".tsv") else ","

    df = pd.read_csv(path, sep=sep)

    refs = df["target"].fillna("").astype(str).tolist()
    hyps = df["prediction"].fillna("").astype(str).tolist()

    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    refs = [r for r, h in valid]
    hyps = [h for r, h in valid]

    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    print(f"BLEU score: {bleu.score:.4f}")
    print(bleu.format())

    # SAS
    model = SentenceTransformer(sas_model_name)

    ref_emb = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    hyp_emb = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)

    similarities = cos_sim(hyp_emb, ref_emb).diagonal()

    sas_mean = float(similarities.mean())
    sas_min = float(similarities.min())
    sas_max = float(similarities.max())

    print(f"SAS mean: {sas_mean:.4f}")
    print(f"SAS min: {sas_min:.4f}")
    print(f"SAS max: {sas_max:.4f}")

    return {
        "bleu": bleu,
        "sas_mean": sas_mean,
        "sas_min": sas_min,
        "sas_max": sas_max,
        "sas_scores": similarities.cpu().tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/dev/stdin")
    parser.add_argument("--sep", default=",")
    parser.add_argument(
        "--sas-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    args = parser.parse_args()

    compute_metrics(args.file, sep=args.sep, sas_model_name=args.sas_model)