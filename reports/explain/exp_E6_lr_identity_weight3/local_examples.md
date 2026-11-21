# Local explanations: exp_E6_lr_identity_weight3

Contribution of each word to the logit (tf-idf value x coefficient, summed over the word's features).
Positive pushes toward the label. Contributions + intercept = logit, exactly. Profanity is masked.

### template sentence

> I am a gay person

Scores: toxic 0.971 (cutoff 0.503), identity_hate 0.799 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | +0.27 | -0.75 |
| a | +0.00 | +0.00 |
| **gay** | +4.47 | +5.50 |
| person | +0.41 | -0.39 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | +3.50 | +1.38 |

### template sentence

> I am a homosexual person

Scores: toxic 0.828 (cutoff 0.503), identity_hate 0.972 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.15 | -0.12 |
| a | +0.00 | +0.00 |
| **homosexual** | +2.81 | +7.19 |
| **person** | +0.57 | -0.54 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | +1.57 | +3.56 |

### template sentence

> I am a proud person

Scores: toxic 0.084 (cutoff 0.503), identity_hate 0.011 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.44 | -0.53 |
| a | +0.00 | +0.00 |
| **proud** | -0.72 | -0.60 |
| person | +0.42 | -0.40 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | -2.39 | -4.50 |

### template sentence

> I am a muslim person

Scores: toxic 0.255 (cutoff 0.503), identity_hate 0.588 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.16 | -0.13 |
| a | +0.00 | +0.00 |
| muslim | +0.13 | +4.04 |
| **person** | +0.61 | -0.59 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | -1.07 | +0.35 |

### template sentence

> You are a great actor

Scores: toxic 0.641 (cutoff 0.503), identity_hate 0.162 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| **You** | +1.85 | +0.48 |
| **are** | +1.31 | +1.02 |
| a | +0.00 | +0.00 |
| great | -0.44 | -0.27 |
| actor | -0.49 | +0.09 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | +0.58 | -1.64 |

### false positive (toxic)

> Thabk you

Scores: toxic 1.000 (cutoff 0.503), identity_hate 0.036 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| Thabk | +0.00 | +0.00 |
| **you** | +9.92 | -0.32 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | +8.26 | -3.29 |

### false positive (toxic)

> ::::Thnk you.

Scores: toxic 1.000 (cutoff 0.503), identity_hate 0.036 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| Thnk | +0.00 | +0.00 |
| **you** | +9.92 | -0.32 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | +8.26 | -3.29 |

### false negative (toxic)

> One of the ugliest politicians in Europe.

Scores: toxic 0.018 (cutoff 0.503), identity_hate 0.010 (cutoff 0.642)

| word | toxic | identity_hate |
|---|---:|---:|
| One | -0.13 | -0.12 |
| of | -0.32 | -0.43 |
| **the** | -0.65 | -0.57 |
| ugliest | +0.00 | +0.00 |
| politicians | -0.16 | -0.05 |
| **in** | -0.59 | -0.19 |
| **Europe** | -0.52 | -0.23 |
| (other) | +0.00 | +0.00 |
| intercept | -1.66 | -2.97 |
| logit | -4.02 | -4.57 |
