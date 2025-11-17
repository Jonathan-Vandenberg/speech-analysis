# Pronunciation Test Report

_Generated on 2025-11-16T04:15:32.455460Z_

## Approach Summary
- Montreal Forced Aligner validates timing and dictionary coverage.
- Allosaurus phoneme recognition + feature alignment provide per-phoneme scoring.
- Whisper transcription gates scoring so words that were never spoken stay at 0%.
- Random penalties were removed; missing phonemes now score 0 by design.

## Test Matrix
| Test | Category | Type | Overall % | Non-zero Words | Zeroed Words |
| --- | --- | --- | --- | --- | --- |
| basic_match | Basic | match | 61.90 | Hello, world | — |
| basic_mismatch | Basic | mismatch | 39.88 | Hello, world | — |
| consonant_match | Consonant | match | 77.94 | Peter, Piper, picked, a, peck, of, pickled, peppers | — |
| consonant_mismatch | Consonant | mismatch | 27.00 | Peter, Piper, a, peck, pickled, peppers | picked, of |
| vowel_match | Vowel | match | 60.36 | A, unique, audio, of, open, ocean, echoes | — |
| vowel_mismatch | Vowel | mismatch | 10.97 | unique, audio, of, ocean | A, open, echoes |
| long_match | Long | match | 86.96 | In, the, grand, library, of, Alexandria, scholars, gathered, to, debate, complex, problems, while, gentle, breezes, carried, the, scent, of, papyrus | — |
| long_mismatch | Long | mismatch | 21.02 | In, to, complex, problems, while, gentle, the, scent, of, papyrus | the, grand, library, of, Alexandria, scholars, gathered, debate, breezes, carried |
| short_single_match | Short | match | 52.25 | Fox | — |
| short_single_mismatch | Short | mismatch | 18.75 | Fox | — |
| word_twister_match | Twister | match | 86.49 | She, sells, seashells, by, the, seashore | — |
| word_twister_mismatch | Twister | mismatch | 50.89 | She, sells, seashells, by, the, seashore | — |
| th_cluster_match | Cluster | match | 85.47 | This, they, them, these, Thursday | — |
| th_cluster_mismatch | Cluster | mismatch | 37.87 | they, them, these, Thursday | This |
| st_final_match | Cluster | match | 91.00 | Past, fast, last, mast | — |
| st_final_mismatch | Cluster | mismatch | 49.56 | Past, fast, mast | last |
| complicated_match | Complicated | match | 75.54 | Quantum, engineers, synchronize, entangled, particles, to, teleport, encrypted, keys, across, vast, networks | — |
| complicated_mismatch | Complicated | mismatch | 41.94 | Quantum, engineers, entangled, particles, to, teleport, encrypted, keys, across, vast, networks | synchronize |

## JSON Snapshots
```
[
  {
    "name": "basic_match",
    "category": "Basic",
    "type": "match",
    "expected": "Hello world.",
    "actual": "Hello world.",
    "overall": 61.9,
    "non_zero": [
      "Hello",
      "world"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Hello",
            "phonemes": [
              {
                "ipa_label": "h",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "o",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 40.8
          },
          {
            "word_text": "world",
            "phonemes": [
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 35.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 83.0
          }
        ],
        "overall_score": 61.9
      },
      "predicted_text": "Hello world.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "basic_mismatch",
    "category": "Basic",
    "type": "mismatch",
    "expected": "Hello world.",
    "actual": "Goodbye universe.",
    "overall": 39.875,
    "non_zero": [
      "Hello",
      "world"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Hello",
            "phonemes": [
              {
                "ipa_label": "h",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "o",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 35.0
              }
            ],
            "word_score": 51.0
          },
          {
            "word_text": "world",
            "phonemes": [
              {
                "ipa_label": "w",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 28.75
          }
        ],
        "overall_score": 39.875
      },
      "predicted_text": "Hello world.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "consonant_match",
    "category": "Consonant",
    "type": "match",
    "expected": "Peter Piper picked a peck of pickled peppers.",
    "actual": "Peter Piper picked a peck of pickled peppers.",
    "overall": 77.94166666666666,
    "non_zero": [
      "Peter",
      "Piper",
      "picked",
      "a",
      "peck",
      "of",
      "pickled",
      "peppers"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Peter",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 53.25
          },
          {
            "word_text": "Piper",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 95.0
              }
            ],
            "word_score": 80.6
          },
          {
            "word_text": "picked",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 91.75
          },
          {
            "word_text": "a",
            "phonemes": [
              {
                "ipa_label": "e",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 35.0
          },
          {
            "word_text": "peck",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 89.33333333333333
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "v",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "pickled",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 76.4
          },
          {
            "word_text": "peppers",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 98.2
          }
        ],
        "overall_score": 77.94166666666666
      },
      "predicted_text": "Peter Piper picked a peck of pickled peppers.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "consonant_mismatch",
    "category": "Consonant",
    "type": "mismatch",
    "expected": "Peter Piper picked a peck of pickled peppers.",
    "actual": "Betty Botter bought some bitter butter.",
    "overall": 26.995833333333334,
    "non_zero": [
      "Peter",
      "Piper",
      "a",
      "peck",
      "pickled",
      "peppers"
    ],
    "zero_words": [
      "picked",
      "of"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Peter",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 44.99999999999999
              }
            ],
            "word_score": 76.0
          },
          {
            "word_text": "Piper",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 20.0
          },
          {
            "word_text": "picked",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "a",
            "phonemes": [
              {
                "ipa_label": "e",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 7.500000000000001
          },
          {
            "word_text": "peck",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 66.66666666666667
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "v",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "pickled",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 33.8
          },
          {
            "word_text": "peppers",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "z",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 11.999999999999998
          }
        ],
        "overall_score": 26.995833333333334
      },
      "predicted_text": "Peter Piper picked a peck of pickled peppers.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "vowel_match",
    "category": "Vowel",
    "type": "match",
    "expected": "A unique audio of open ocean echoes.",
    "actual": "A unique audio of open ocean echoes.",
    "overall": 60.35714285714285,
    "non_zero": [
      "A",
      "unique",
      "audio",
      "of",
      "open",
      "ocean",
      "echoes"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "A",
            "phonemes": [
              {
                "ipa_label": "e",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 35.0
          },
          {
            "word_text": "unique",
            "phonemes": [
              {
                "ipa_label": "j",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 73.4
          },
          {
            "word_text": "audio",
            "phonemes": [
              {
                "ipa_label": "\u0254",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "o",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 48.0
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "v",
                "phoneme_score": 89.99999999999999
              }
            ],
            "word_score": 67.49999999999999
          },
          {
            "word_text": "open",
            "phonemes": [
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 35.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 66.4
          },
          {
            "word_text": "ocean",
            "phonemes": [
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 73.4
          },
          {
            "word_text": "echoes",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 35.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 89.99999999999999
              }
            ],
            "word_score": 58.8
          }
        ],
        "overall_score": 60.35714285714285
      },
      "predicted_text": "A unique audio of open ocean echoes.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "vowel_mismatch",
    "category": "Vowel",
    "type": "mismatch",
    "expected": "A unique audio of open ocean echoes.",
    "actual": "Under autumn umbrellas we gather quietly.",
    "overall": 10.971428571428572,
    "non_zero": [
      "unique",
      "audio",
      "of",
      "ocean"
    ],
    "zero_words": [
      "A",
      "open",
      "echoes"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "A",
            "phonemes": [
              {
                "ipa_label": "e",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "unique",
            "phonemes": [
              {
                "ipa_label": "j",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 12.0
          },
          {
            "word_text": "audio",
            "phonemes": [
              {
                "ipa_label": "\u0254",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 14.0
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "v",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 15.000000000000002
          },
          {
            "word_text": "open",
            "phonemes": [
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "ocean",
            "phonemes": [
              {
                "ipa_label": "o",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 35.0
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 35.8
          },
          {
            "word_text": "echoes",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "o",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u028a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          }
        ],
        "overall_score": 10.971428571428572
      },
      "predicted_text": "A unique audio of open ocean echoes.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "long_match",
    "category": "Long",
    "type": "match",
    "expected": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
    "actual": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
    "overall": 86.95541666666665,
    "non_zero": [
      "In",
      "the",
      "grand",
      "library",
      "of",
      "Alexandria",
      "scholars",
      "gathered",
      "to",
      "debate",
      "complex",
      "problems",
      "while",
      "gentle",
      "breezes",
      "carried",
      "the",
      "scent",
      "of",
      "papyrus"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "In",
            "phonemes": [
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "grand",
            "phonemes": [
              {
                "ipa_label": "\u0261",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "library",
            "phonemes": [
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "b",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 78.75
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "v",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "Alexandria",
            "phonemes": [
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0261",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i\u0259",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 90.6
          },
          {
            "word_text": "scholars",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 85.0
          },
          {
            "word_text": "gathered",
            "phonemes": [
              {
                "ipa_label": "\u0261",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 82.2
          },
          {
            "word_text": "to",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 35.0
              }
            ],
            "word_score": 67.0
          },
          {
            "word_text": "debate",
            "phonemes": [
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "e",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 30.000000000000004
              }
            ],
            "word_score": 64.83333333333333
          },
          {
            "word_text": "complex",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 95.375
          },
          {
            "word_text": "problems",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "while",
            "phonemes": [
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 77.0
          },
          {
            "word_text": "gentle",
            "phonemes": [
              {
                "ipa_label": "d\u0292",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 48.6
          },
          {
            "word_text": "breezes",
            "phonemes": [
              {
                "ipa_label": "b",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 88.5
          },
          {
            "word_text": "carried",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "scent",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 91.75
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "v",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "papyrus",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0250",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 77.5
          }
        ],
        "overall_score": 86.95541666666665
      },
      "predicted_text": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "long_mismatch",
    "category": "Long",
    "type": "mismatch",
    "expected": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
    "actual": "Modern satellites orbit Earth while engineers in mission control whisper about telemetry and solar winds.",
    "overall": 21.025,
    "non_zero": [
      "In",
      "to",
      "complex",
      "problems",
      "while",
      "gentle",
      "the",
      "scent",
      "of",
      "papyrus"
    ],
    "zero_words": [
      "the",
      "grand",
      "library",
      "of",
      "Alexandria",
      "scholars",
      "gathered",
      "debate",
      "breezes",
      "carried"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "In",
            "phonemes": [
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 35.0
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "grand",
            "phonemes": [
              {
                "ipa_label": "\u0261",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "library",
            "phonemes": [
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "v",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "Alexandria",
            "phonemes": [
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0261",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "i\u0259",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "scholars",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "gathered",
            "phonemes": [
              {
                "ipa_label": "\u0261",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "to",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 30.0
          },
          {
            "word_text": "debate",
            "phonemes": [
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "e",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "complex",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "m",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 29.375
          },
          {
            "word_text": "problems",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "b",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 30.625
          },
          {
            "word_text": "while",
            "phonemes": [
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 64.5
          },
          {
            "word_text": "gentle",
            "phonemes": [
              {
                "ipa_label": "d\u0292",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 48.0
          },
          {
            "word_text": "breezes",
            "phonemes": [
              {
                "ipa_label": "b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "carried",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 44.99999999999999
              }
            ],
            "word_score": 42.5
          },
          {
            "word_text": "scent",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "n",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 48.75
          },
          {
            "word_text": "of",
            "phonemes": [
              {
                "ipa_label": "\u028c",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "v",
                "phoneme_score": 40.0
              }
            ],
            "word_score": 42.5
          },
          {
            "word_text": "papyrus",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u0250",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "p",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "a",
                "phoneme_score": 35.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "s",
                "phoneme_score": 89.99999999999999
              }
            ],
            "word_score": 49.25
          }
        ],
        "overall_score": 21.025
      },
      "predicted_text": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "short_single_match",
    "category": "Short",
    "type": "match",
    "expected": "Fox.",
    "actual": "Fox.",
    "overall": 52.25,
    "non_zero": [
      "Fox"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Fox",
            "phonemes": [
              {
                "ipa_label": "f",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 52.25
          }
        ],
        "overall_score": 52.25
      },
      "predicted_text": "Fox.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "short_single_mismatch",
    "category": "Short",
    "type": "mismatch",
    "expected": "Fox.",
    "actual": "Dog.",
    "overall": 18.75,
    "non_zero": [
      "Fox"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Fox",
            "phonemes": [
              {
                "ipa_label": "f",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 18.75
          }
        ],
        "overall_score": 18.75
      },
      "predicted_text": "Fox.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "word_twister_match",
    "category": "Twister",
    "type": "match",
    "expected": "She sells seashells by the seashore.",
    "actual": "She sells seashells by the seashore.",
    "overall": 86.48611111111113,
    "non_zero": [
      "She",
      "sells",
      "seashells",
      "by",
      "the",
      "seashore"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "She",
            "phonemes": [
              {
                "ipa_label": "\u0283",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "sells",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 74.25
          },
          {
            "word_text": "seashells",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "by",
            "phonemes": [
              {
                "ipa_label": "b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 69.66666666666667
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "seashore",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0254r",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 78.0
          }
        ],
        "overall_score": 86.48611111111113
      },
      "predicted_text": "She sells seashells by the seashore.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "word_twister_mismatch",
    "category": "Twister",
    "type": "mismatch",
    "expected": "She sells seashells by the seashore.",
    "actual": "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
    "overall": 50.888888888888886,
    "non_zero": [
      "She",
      "sells",
      "seashells",
      "by",
      "the",
      "seashore"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "She",
            "phonemes": [
              {
                "ipa_label": "\u0283",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 50.0
          },
          {
            "word_text": "sells",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 62.5
          },
          {
            "word_text": "seashells",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "i",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 26.666666666666668
          },
          {
            "word_text": "by",
            "phonemes": [
              {
                "ipa_label": "b",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 51.666666666666664
          },
          {
            "word_text": "the",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 79.5
          },
          {
            "word_text": "seashore",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0283",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u0254r",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 35.0
          }
        ],
        "overall_score": 50.888888888888886
      },
      "predicted_text": "She sells seashells by the seashore.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "th_cluster_match",
    "category": "Cluster",
    "type": "match",
    "expected": "This they them these Thursday.",
    "actual": "This they them these Thursday.",
    "overall": 85.46666666666667,
    "non_zero": [
      "This",
      "they",
      "them",
      "these",
      "Thursday"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "This",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "they",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "e",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 69.66666666666667
          },
          {
            "word_text": "them",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "these",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "Thursday",
            "phonemes": [
              {
                "ipa_label": "\u03b8",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "d",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "e",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 60.666666666666664
          }
        ],
        "overall_score": 85.46666666666667
      },
      "predicted_text": "This they them these Thursday.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "th_cluster_mismatch",
    "category": "Cluster",
    "type": "mismatch",
    "expected": "This they them these Thursday.",
    "actual": "Random words without the th cluster present anywhere.",
    "overall": 37.86666666666667,
    "non_zero": [
      "they",
      "them",
      "these",
      "Thursday"
    ],
    "zero_words": [
      "This"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "This",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "they",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "e",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 15.000000000000002
          },
          {
            "word_text": "them",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 40.0
              }
            ],
            "word_score": 69.66666666666667
          },
          {
            "word_text": "these",
            "phonemes": [
              {
                "ipa_label": "\u00f0",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 76.33333333333333
          },
          {
            "word_text": "Thursday",
            "phonemes": [
              {
                "ipa_label": "\u03b8",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "d",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "e",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 28.333333333333332
          }
        ],
        "overall_score": 37.86666666666667
      },
      "predicted_text": "This they them these Thursday.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "st_final_match",
    "category": "Cluster",
    "type": "match",
    "expected": "Past fast last mast.",
    "actual": "Past fast last mast.",
    "overall": 91.0,
    "non_zero": [
      "Past",
      "fast",
      "last",
      "mast"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Past",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "fast",
            "phonemes": [
              {
                "ipa_label": "f",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 74.25
          },
          {
            "word_text": "last",
            "phonemes": [
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "mast",
            "phonemes": [
              {
                "ipa_label": "m",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 91.75
          }
        ],
        "overall_score": 91.0
      },
      "predicted_text": "Past fast last mast.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "st_final_mismatch",
    "category": "Cluster",
    "type": "mismatch",
    "expected": "Past fast last mast.",
    "actual": "These phrases avoid the sharp st ending altogether.",
    "overall": 49.5625,
    "non_zero": [
      "Past",
      "fast",
      "mast"
    ],
    "zero_words": [
      "last"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Past",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 65.0
          },
          {
            "word_text": "fast",
            "phonemes": [
              {
                "ipa_label": "f",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 89.5
          },
          {
            "word_text": "last",
            "phonemes": [
              {
                "ipa_label": "l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "mast",
            "phonemes": [
              {
                "ipa_label": "m",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 15.000000000000002
              }
            ],
            "word_score": 43.75
          }
        ],
        "overall_score": 49.5625
      },
      "predicted_text": "Past fast last mast.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "complicated_match",
    "category": "Complicated",
    "type": "match",
    "expected": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
    "actual": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
    "overall": 75.5373015873016,
    "non_zero": [
      "Quantum",
      "engineers",
      "synchronize",
      "entangled",
      "particles",
      "to",
      "teleport",
      "encrypted",
      "keys",
      "across",
      "vast",
      "networks"
    ],
    "zero_words": [],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Quantum",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0254",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "m",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 70.85714285714286
          },
          {
            "word_text": "engineers",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "d\u0292",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026ar",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 56.857142857142854
          },
          {
            "word_text": "synchronize",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 95.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 89.99999999999999
              }
            ],
            "word_score": 81.0
          },
          {
            "word_text": "entangled",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0261",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 65.25
          },
          {
            "word_text": "particles",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0251r",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 89.99999999999999
              }
            ],
            "word_score": 69.57142857142857
          },
          {
            "word_text": "to",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 35.0
              }
            ],
            "word_score": 67.0
          },
          {
            "word_text": "teleport",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "l",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0254r",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 73.14285714285714
          },
          {
            "word_text": "encrypted",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 72.33333333333333
          },
          {
            "word_text": "keys",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 99.0
          },
          {
            "word_text": "across",
            "phonemes": [
              {
                "ipa_label": "\u0259",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 80.4
          },
          {
            "word_text": "vast",
            "phonemes": [
              {
                "ipa_label": "v",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 91.75
          },
          {
            "word_text": "networks",
            "phonemes": [
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 79.28571428571429
          }
        ],
        "overall_score": 75.5373015873016
      },
      "predicted_text": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  },
  {
    "name": "complicated_mismatch",
    "category": "Complicated",
    "type": "mismatch",
    "expected": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
    "actual": "Gardeners quietly trim bonsai trees while rain taps rhythmically against the greenhouse glass.",
    "overall": 41.93832671957672,
    "non_zero": [
      "Quantum",
      "engineers",
      "entangled",
      "particles",
      "to",
      "teleport",
      "encrypted",
      "keys",
      "across",
      "vast",
      "networks"
    ],
    "zero_words": [
      "synchronize"
    ],
    "payload": {
      "pronunciation": {
        "words": [
          {
            "word_text": "Quantum",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "w",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0254",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 89.99999999999999
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "m",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 69.71428571428571
          },
          {
            "word_text": "engineers",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "n",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "d\u0292",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026ar",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 24.142857142857142
          },
          {
            "word_text": "synchronize",
            "phonemes": [
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 0.0
          },
          {
            "word_text": "entangled",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "n",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0261",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 21.125
          },
          {
            "word_text": "particles",
            "phonemes": [
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0251r",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "r",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0259l",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "z",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 34.285714285714285
          },
          {
            "word_text": "to",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "u",
                "phoneme_score": 35.0
              }
            ],
            "word_score": 52.5
          },
          {
            "word_text": "teleport",
            "phonemes": [
              {
                "ipa_label": "t",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "l",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "p",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u0254r",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 25.714285714285715
          },
          {
            "word_text": "encrypted",
            "phonemes": [
              {
                "ipa_label": "\u025b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "\u014b",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u026a",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "p",
                "phoneme_score": 30.000000000000004
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              },
              {
                "ipa_label": "\u1d7b",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "d",
                "phoneme_score": 70.0
              }
            ],
            "word_score": 41.111111111111114
          },
          {
            "word_text": "keys",
            "phonemes": [
              {
                "ipa_label": "k",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "i",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "z",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 79.66666666666667
          },
          {
            "word_text": "across",
            "phonemes": [
              {
                "ipa_label": "\u0259",
                "phoneme_score": 0.0
              },
              {
                "ipa_label": "k",
                "phoneme_score": 40.0
              },
              {
                "ipa_label": "r",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "\u0251",
                "phoneme_score": 44.99999999999999
              },
              {
                "ipa_label": "s",
                "phoneme_score": 0.0
              }
            ],
            "word_score": 31.0
          },
          {
            "word_text": "vast",
            "phonemes": [
              {
                "ipa_label": "v",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "\u00e6",
                "phoneme_score": 70.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "t",
                "phoneme_score": 60.0
              }
            ],
            "word_score": 61.0
          },
          {
            "word_text": "networks",
            "phonemes": [
              {
                "ipa_label": "n",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025b",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "t",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "w",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "\u025c",
                "phoneme_score": 15.000000000000002
              },
              {
                "ipa_label": "k",
                "phoneme_score": 99.0
              },
              {
                "ipa_label": "s",
                "phoneme_score": 99.0
              }
            ],
            "word_score": 63.0
          }
        ],
        "overall_score": 41.93832671957672
      },
      "predicted_text": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
      "metrics": null,
      "grammar": null,
      "relevance": null,
      "ielts_score": null
    }
  }
]
```