# Local explanations: exp_E3_lr_balanced

Contribution of each word to the logit (tf-idf value x coefficient, summed over the word's features).
Positive pushes toward the label. Contributions + intercept = logit, exactly. Profanity is masked.

### template sentence

> I am a gay person

Scores: toxic 0.976 (cutoff 0.524), identity_hate 0.846 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | +0.25 | -0.61 |
| a | +0.00 | +0.00 |
| **gay** | +4.79 | +5.74 |
| person | +0.39 | -0.39 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | +3.70 | +1.70 |

### template sentence

> I am a homosexual person

Scores: toxic 0.889 (cutoff 0.524), identity_hate 0.988 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.06 | +0.11 |
| a | +0.00 | +0.00 |
| **homosexual** | +3.33 | +7.89 |
| **person** | +0.53 | -0.53 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | +2.08 | +4.43 |

### template sentence

> I am a proud person

Scores: toxic 0.119 (cutoff 0.524), identity_hate 0.040 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.22 | +0.04 |
| a | +0.00 | +0.00 |
| proud | -0.44 | +0.23 |
| person | +0.39 | -0.39 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | -2.00 | -3.17 |

### template sentence

> I am a muslim person

Scores: toxic 0.444 (cutoff 0.524), identity_hate 0.740 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.00 | +0.00 |
| am | -0.07 | +0.12 |
| a | +0.00 | +0.00 |
| **muslim** | +0.99 | +4.55 |
| **person** | +0.58 | -0.57 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | -0.22 | +1.05 |

### template sentence

> You are a great actor

Scores: toxic 0.632 (cutoff 0.524), identity_hate 0.149 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| **You** | +1.79 | +0.47 |
| **are** | +1.31 | +1.03 |
| a | +0.00 | +0.00 |
| great | -0.37 | -0.24 |
| actor | -0.46 | +0.05 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | +0.54 | -1.74 |

### false positive (toxic)

> Thabk you

Scores: toxic 1.000 (cutoff 0.524), identity_hate 0.035 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| Thabk | +0.00 | +0.00 |
| **you** | +9.67 | -0.28 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | +7.94 | -3.32 |

### false positive (toxic)

> ::::Thnk you.

Scores: toxic 1.000 (cutoff 0.524), identity_hate 0.035 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| Thnk | +0.00 | +0.00 |
| **you** | +9.67 | -0.28 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | +7.94 | -3.32 |

### false negative (toxic)

> One of the ugliest politicians in Europe.

Scores: toxic 0.021 (cutoff 0.524), identity_hate 0.012 (cutoff 0.599)

| word | toxic | identity_hate |
|---|---:|---:|
| One | -0.11 | -0.09 |
| of | -0.27 | -0.34 |
| **the** | -0.63 | -0.52 |
| ugliest | +0.00 | +0.00 |
| politicians | -0.17 | -0.06 |
| **in** | -0.52 | -0.14 |
| Europe | -0.42 | -0.19 |
| (other) | +0.00 | +0.00 |
| intercept | -1.73 | -3.05 |
| logit | -3.86 | -4.39 |
