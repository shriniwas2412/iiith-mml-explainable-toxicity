"""Download the public files the bias audit is built from (about 3.3 MB).

  data/external/dixon/   sentence templates released with Dixon et al. 2018
                         (conversationai/unintended-ml-bias-analysis, CC BY 4.0)
  data/external/india/   Indian identity term lists released with Bhatt et al. 2022
                         (google-research-datasets/nlp-fairness-for-india, Apache-2.0)

    python scripts/07_fetch_bias_resources.py
    python scripts/07_fetch_bias_resources.py --hindi     # also the romanised Hindi templates
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DIXON = "https://raw.githubusercontent.com/conversationai/unintended-ml-bias-analysis/main/sentence_templates/"
INDIA = "https://raw.githubusercontent.com/google-research-datasets/nlp-fairness-for-india/main/"

FILES = {
    "dixon/en_sentence_templates.csv": DIXON + "en_sentence_templates.csv",
    "india/caste_idterms.tsv": INDIA + "caste_idterms.tsv",
    "india/religion_idterms.tsv": INDIA + "religion_idterms.tsv",
    "india/region_idterms.tsv": INDIA + "region_idterms.tsv",
}
HINDI = {"dixon/hi-Latn_sentence_templates.csv": DIXON + "hi-Latn_sentence_templates.csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/external")
    parser.add_argument("--hindi", action="store_true", help="Also fetch the hi-Latn templates")
    args = parser.parse_args()

    files = {**FILES, **(HINDI if args.hindi else {})}
    for relative, url in files.items():
        target = Path(args.out) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"{relative}: already present")
            continue
        urllib.request.urlretrieve(url, target)
        print(f"{relative}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
