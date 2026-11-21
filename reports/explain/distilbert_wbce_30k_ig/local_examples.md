# Local explanations (Integrated Gradients): distilbert_wbce_30k_ig

Attribution of each word to the logit, 50 steps, baseline = PAD embeddings with CLS/SEP kept,
word pieces summed per word. Positive pushes toward the label. Attributions add up to logit minus baseline logit
up to the integration error shown in the last row. Profanity is masked.

### template sentence

> I am a gay person

Scores: toxic 0.843 (cutoff 0.792), identity_hate 0.696 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.02 | -0.00 |
| am | -0.39 | -0.31 |
| **a** | +1.51 | +1.45 |
| **gay** | +3.61 | +4.41 |
| **person** | +0.81 | +1.03 |
| sum of attributions | +5.547 | +6.581 |
| logit | +1.683 | +0.828 |
| baseline logit | -3.878 | -5.767 |
| completeness error | -0.014 | -0.014 |

### template sentence

> I am a homosexual person

Scores: toxic 0.876 (cutoff 0.792), identity_hate 0.765 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | -0.02 | -0.01 |
| am | -0.35 | -0.25 |
| **a** | +2.01 | +2.00 |
| **homosexual** | +3.16 | +4.02 |
| **person** | +1.04 | +1.18 |
| sum of attributions | +5.848 | +6.940 |
| logit | +1.953 | +1.181 |
| baseline logit | -3.878 | -5.767 |
| completeness error | +0.017 | -0.008 |

### template sentence

> I am a proud person

Scores: toxic 0.048 (cutoff 0.792), identity_hate 0.007 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.18 | +0.15 |
| **am** | -0.54 | -0.25 |
| **a** | +0.74 | +0.39 |
| proud | -0.50 | -0.22 |
| **person** | +0.99 | +0.69 |
| sum of attributions | +0.876 | +0.751 |
| logit | -2.996 | -5.007 |
| baseline logit | -3.878 | -5.767 |
| completeness error | -0.006 | -0.009 |

### template sentence

> I am a muslim person

Scores: toxic 0.242 (cutoff 0.792), identity_hate 0.050 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.13 | +0.15 |
| **am** | -0.51 | -0.26 |
| **a** | +1.26 | +0.71 |
| **muslim** | +0.54 | +1.19 |
| **person** | +1.32 | +1.04 |
| sum of attributions | +2.746 | +2.827 |
| logit | -1.143 | -2.949 |
| baseline logit | -3.878 | -5.767 |
| completeness error | +0.011 | +0.009 |

### template sentence

> You are a great actor

Scores: toxic 0.127 (cutoff 0.792), identity_hate 0.010 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| **You** | +0.86 | +0.53 |
| are | +0.37 | +0.16 |
| **a** | +0.68 | +0.21 |
| great | -0.33 | -0.34 |
| actor | +0.38 | +0.58 |
| sum of attributions | +1.951 | +1.142 |
| logit | -1.926 | -4.627 |
| baseline logit | -3.878 | -5.767 |
| completeness error | +0.000 | +0.002 |

### template sentence

> I am a dalit person

Scores: toxic 0.860 (cutoff 0.792), identity_hate 0.282 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.31 | +0.44 |
| am | -0.25 | -0.23 |
| **a** | +1.33 | +0.83 |
| **dalit** | +2.42 | +2.61 |
| **person** | +1.52 | +0.98 |
| sum of attributions | +5.334 | +4.625 |
| logit | +1.816 | -0.935 |
| baseline logit | -3.514 | -5.553 |
| completeness error | +0.004 | +0.007 |

### template sentence

> I am an obc person

Scores: toxic 0.913 (cutoff 0.792), identity_hate 0.198 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.18 | +0.15 |
| am | -0.11 | -0.10 |
| **an** | +1.31 | +0.86 |
| **obc** | +2.69 | +2.09 |
| **person** | +1.79 | +1.15 |
| sum of attributions | +5.855 | +4.145 |
| logit | +2.346 | -1.399 |
| baseline logit | -3.514 | -5.553 |
| completeness error | -0.006 | -0.008 |

### template sentence

> I am a brahmin person

Scores: toxic 0.165 (cutoff 0.792), identity_hate 0.030 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.24 | +0.29 |
| **am** | -0.61 | -0.31 |
| **a** | +1.19 | +0.57 |
| brahmin | -0.25 | +0.64 |
| **person** | +0.88 | +0.53 |
| sum of attributions | +1.444 | +1.727 |
| logit | -1.621 | -3.486 |
| baseline logit | -3.055 | -5.205 |
| completeness error | +0.009 | +0.008 |

### template sentence

> I am a bihari person

Scores: toxic 0.095 (cutoff 0.792), identity_hate 0.015 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| I | +0.24 | +0.26 |
| am | -0.50 | -0.23 |
| **a** | +1.11 | +0.47 |
| bihari | -0.44 | +0.18 |
| **person** | +0.85 | +0.64 |
| sum of attributions | +1.265 | +1.335 |
| logit | -2.250 | -4.214 |
| baseline logit | -3.514 | -5.553 |
| completeness error | +0.001 | -0.004 |

### false positive (toxic)

> == Nerd == Materialnerd! Get out of your mother's basement and get a life-leave me the hell alone. I suggest you go suck off Barek Obama bin Laden and Fagreterion.

Scores: toxic 0.989 (cutoff 0.792), identity_hate 0.383 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| Nerd | +0.29 | +0.31 |
| ! | +0.32 | +0.18 |
| your | +0.46 | +0.42 |
| mother | +0.47 | +0.60 |
| hell | +0.38 | +0.13 |
| you | +0.33 | +0.26 |
| go | +0.33 | +0.31 |
| **suck** | +0.87 | +0.76 |
| sum of attributions | +6.074 | +3.842 |
| logit | +4.477 | -0.477 |
| baseline logit | -1.616 | -4.309 |
| completeness error | -0.019 | +0.010 |

### false positive (toxic)

> Off the dope man Jumpsteady kicks ass. Anybody know his finisher as Evil Dead in JCW? MMFCL!!

Scores: toxic 0.989 (cutoff 0.792), identity_hate 0.586 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| dope | +0.38 | +0.45 |
| **man** | +0.52 | +0.53 |
| Jumpsteady | +0.38 | +0.26 |
| **ass** | +1.05 | +0.76 |
| Evil | +0.49 | +0.61 |
| Dead | +0.35 | +0.49 |
| ! | +0.33 | +0.14 |
| ! | +0.36 | +0.20 |
| sum of attributions | +5.836 | +4.417 |
| logit | +4.460 | +0.348 |
| baseline logit | -1.378 | -4.072 |
| completeness error | -0.002 | -0.003 |

### false positive (toxic)

> U.S.A U.S.A U.S.A COULD KICK YOUR DUMB ASSES ANY DAY LOL

Scores: toxic 0.988 (cutoff 0.792), identity_hate 0.532 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| U | +0.13 | +0.14 |
| U | +0.15 | +0.13 |
| **KICK** | +0.70 | +0.55 |
| **YOUR** | +1.41 | +1.18 |
| **DUMB** | +1.56 | +1.69 |
| **ASSES** | +0.72 | +0.52 |
| DAY | +0.17 | +0.13 |
| LOL | +0.39 | +0.47 |
| sum of attributions | +5.735 | +4.054 |
| logit | +4.417 | +0.130 |
| baseline logit | -1.313 | -3.931 |
| completeness error | +0.005 | -0.007 |

### false negative (toxic)

> " :I would direct you to WP:ERA, but you are already well aware of that and have just been blocked as a sockpuppet of .- "

Scores: toxic 0.005 (cutoff 0.792), identity_hate 0.001 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| you | +0.27 | +0.04 |
| to | -0.40 | -0.13 |
| , | -0.27 | -0.07 |
| you | +0.44 | +0.10 |
| that | -0.47 | -0.16 |
| been | -0.30 | -0.12 |
| **sockpuppet** | +0.52 | -0.02 |
| " | -0.32 | -0.13 |
| sum of attributions | -3.779 | -2.343 |
| logit | -5.256 | -6.548 |
| baseline logit | -1.478 | -4.206 |
| completeness error | -0.001 | -0.002 |

### false negative (toxic)

> ::::That's the silliest thing that I've ever heard.

Scores: toxic 0.005 (cutoff 0.792), identity_hate 0.001 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| : | -0.21 | -0.19 |
| That | -0.44 | -0.23 |
| s | -0.37 | -0.21 |
| the | -0.45 | -0.20 |
| silliest | -0.30 | -0.26 |
| **that** | -0.72 | -0.32 |
| ' | -0.26 | -0.17 |
| . | -0.33 | -0.30 |
| sum of attributions | -3.801 | -2.644 |
| logit | -5.203 | -6.550 |
| baseline logit | -1.411 | -3.909 |
| completeness error | -0.009 | -0.003 |

### false negative (toxic)

> Using your finger or a paper towel to rub in the penicl, is always a good idea when trying to shade.

Scores: toxic 0.007 (cutoff 0.792), identity_hate 0.002 (cutoff 0.777)

| word | toxic | identity_hate |
|---|---:|---:|
| **Using** | -0.70 | -0.24 |
| **your** | +1.07 | +0.32 |
| **finger** | +0.65 | +0.07 |
| **paper** | -0.66 | -0.47 |
| **towel** | +0.58 | +0.02 |
| to | -0.50 | -0.11 |
| the | -0.47 | -0.17 |
| **idea** | -0.60 | -0.25 |
| sum of attributions | -3.644 | -2.526 |
| logit | -4.955 | -6.456 |
| baseline logit | -1.313 | -3.931 |
| completeness error | -0.002 | -0.001 |
