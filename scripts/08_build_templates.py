"""Build the matched template set for the bias audit.

Every frame from the Dixon et al. templates is filled with (a) the original 50
identity terms, (b) Indian caste, religion and region terms and (c) a neutral
word, so sentences differ only in the identity term.

    python scripts/08_build_templates.py
    python scripts/08_build_templates.py --max-name-frames 50    # smaller set for quick runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toxicity.templates import build_template_set

RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dixon", default="data/external/dixon/en_sentence_templates.csv")
    parser.add_argument("--india-dir", default="data/external/india")
    parser.add_argument("--out", default="data/templates/templates_en.csv")
    parser.add_argument("--max-name-frames", type=int, default=None,
                        help="Keep this many '<name> is a <adjective> <identity>' frames per toxicity class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    templates = build_template_set(
        args.dixon,
        args.india_dir,
        overrides_path=RESOURCES / "india_term_overrides.csv",
        dixon_axes_path=RESOURCES / "dixon_term_axes.csv",
        max_name_frames=args.max_name_frames,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    templates.to_csv(out, index=False)

    print(templates.groupby(["term_set", "toxicity"]).size().unstack(fill_value=0))
    terms = templates[templates["identity"] != ""].groupby(["term_set", "axis"])["identity"].nunique()
    print("\nIdentity terms per group:")
    print(terms)
    print(f"\n{len(templates):,} sentences written to {out}")


if __name__ == "__main__":
    main()
