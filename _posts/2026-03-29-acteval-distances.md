---
layout: post
title: Measuring the Distance Between Activity Populations
description: >
  A walkthrough of acteval's semantic distance pipeline: how to extract
  meaningful features from daily activity schedules, compute Earth Mover's
  Distance between populations, and aggregate the results into interpretable
  domain scores.
date: 2026-03-29
tags: acteval, transport, evaluation
categories: Dev
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

## Acteval

The following is a quick primer for <a href="https://github.com/fredshone/acteval">acteval</a> - a project for measuring the difference between populations of activity schedules.

A tyical use case for acteval is to check how well you modelled or synthesised schedules match some target distribution.

## What are activity sequences?

Activity schedules are the things we do and when. Like leave the house at 7, work from 8.30 till 5.30, then nip to the shops for 20 minutes on the way home.

More formally, we define an activity schedule as a sequence of non-overlapping episodes that typically span a 24-hour day. Each episode records who did it (we use person ID - `pid`), what activity they performed (`act`), and when (`start`, `end` and/or `duration`):

```python
import pandas as pd
from acteval.describe import plot

example = pd.DataFrame([
    {"pid": 0, "act": "home",    "start":    0, "end":  450},
    {"pid": 0, "act": "work",    "start":  450, "end":  810},
    {"pid": 0, "act": "eat_out", "start":  810, "end":  870},
    {"pid": 0, "act": "work",    "start":  870, "end": 1080},
    {"pid": 0, "act": "home",    "start": 1080, "end": 1440},
])

ACTS = {  # activity type to colour mapping
    "home": "#5b8dd9",
    "work": "#d95b5b",
    "eat_out": "#e8a838",
    "shop": "#5db85d",
    "leisure": "#a05bb8",
    "education": "#5bbcbc",
}

plot.gantt(
    populations={"Sample": example},
    act_colors=ACTS,
    acts=ACTS.keys(),
)
```

<div class="text-center">
{% include figure.liquid path="assets/img/acteval/fig1-gantt.svg" caption="Example activity schedule." max-width="90%" %}
</div>

Start and end times are typically minutes from midnight but for now any consistent unit works; the library normalises internally.

## What sort of distances?

On it's own a schedule is a complex object. As a thought experiment we can image representing a schedule as 1440 activity choices (each representing a minute of the day). If we were to represent a schedule with just two possible activity types (say `home` and `work`) then there are $2^{1440}$ possible schedules. A population is then comprised of potentially millions of person's schedules.

```python
import pandas as pd
from acteval.describe import plot
from acteval.scripts.generate_blog_plots import generate_suburban_workers, generate_lifestyle

A = generate_suburban_workers(1000)

plot.gantt(populations={"A": A}, act_colors=ACTS, acts=ACTS.keys())

```

<div class="text-center">
{% include figure.liquid path="assets/img/acteval/fig2-gantt.svg" caption="Example activity schedule." max-width="90%" %}
</div>

We can think of these populations as really big complex distributions, with mixtures of discrete and continuous dimensions. Vast swathes of this theorized disribution are empty (no one *should* flip-flop hundreds of times between work and home in a day) and some are quite dense (like the classic home-work-home sequence).

Our focus is to pick out subtle changes, including changes to pattens in activity particapations, changes to the order of activities, their durations and when they happen.

To do this we try to measure differences in a meaningful way. Specifically, we approximate the nasty complex distribution of the population with loads and loads of meaningful marginal distributions. For example:

- how often do people participate in `home`, `work`, `shop`, and so on?
- how often do people travel from `work` to `shop`, `shop` to `work` and so on?
- when do people tend to start `work` and how long do they `work` for?

We tend to refer to these as population features, rather than marginal distributions.

## Features

We expose population features as pre-computed numpy arrays via the `eval.Population` class. For example, we can take a look at the number of times each activiy type occurs in each schedule:

```python
from acteval import Population
from acteval.describe import plot

A = Population(generate_urban_workers(1000))
print(A.count_matrix[-3:])
# [
#  [0 1 2 1 0 0]
#  [0 0 2 0 1 1]
#  [0 0 2 0 0 1]
# ]
print(A.int_to_act)
# ['eat_out' 'education' 'home' 'leisure' 'shop' 'work']

```

---

`Population` integer-encodes activities and person IDs on construction, and lazily caches expensive derived quantities (n-gram keys, count matrices) on first access. This pays off when evaluating many synthetic models against the same observed data.

## Feature distributions

For a sample or population of activity schedules, these features form distributions. For the following we will consider a nice simple feature and it's distribution, the number of activities in each plan (i.e. it's length):

```python
from acteval.features.participation import sequence_lengths
from acteval.describe import plot

A = urban_workers(1000)

print(sequence_lengths(Population(A)).aggregate())
# {'sequence lengths': (array([3., 4., 5.]), array([230, 395, 375]))}

_ = plot.sequence_lengths({"A": A})

```

<div class="text-center">
{% include figure.liquid path="assets/img/acteval/fig3-lengths.svg" caption="Example population feature distributions." max-width="50%" %}
</div>

Note that `sequence_lengths_per_pid` returns a `PidFeatures` object. We then use `aggregate` to extract the distribution as a tuple of counts and their frequncies. `PidFeatures` can be subset based on some sub-population of person ids to get more refined distributions. But more on this later.

## Feature distances

Consider two distribution of sequence lengths, one from population `A`, the other from population `B`:

```python
from acteval.features.participation import sequence_lengths
from acteval.describe import plot

A = urban_workers(1000)
B = leisure_dominant(1000)

print(sequence_lengths(Population(A)).aggregate())
# {'sequence lengths': (array([3., 4., 5.]), array([230, 395, 375]))}

print(sequence_lengths(Population(B)).aggregate())
# {'sequence lengths': (array([3., 4., 5.]), array([407, 520,  73]))}

_ = plot.sequence_lengths({"A": A, "B": B})
```

<div class="text-center">
{% include figure.liquid path="assets/img/acteval/fig4-lengths.svg" caption="Example population feature distributions." max-width="50%" %}
</div>


We measure the distance between these two distributions using **Earth Mover's Distance** (EMD), also known as the Wasserstein distance. Informally: imagine each distribution as a pile of soil spread across a number line. The EMD is the minimum amount of work needed to rearrange one pile into the shape of the other, where work = area × distance moved.


```python

from acteval.distance.wasserstein import emd
from acteval.features.participation import sequence_lengths

A = urban_workers(1000)
B = leisure_dominant(1000)

features_A = sequence_lengths(Population(A)).aggregate()
features_B = sequence_lengths(Population(B)).aggregate()

print(emd(features_A["sequence lengths"], features_B["sequence lengths"]))
# 0.47899999999999987

```

We like EMD because the unit of distance is often quite meaningful. For example, a distance of 1, between sequence length distributions, is equivalent to saying that a population's schedules are typically one activitiy longer or shorter than  another. But keep in mind they could also have the same expected value, but be distributed more and less flatly. To find out, a user has to plot the distribution or calculate descriptive metrics.

```python
from acteval.distance.wasserstein import emd

A = urban_workers(1000)
B = leisure_dominant(1000)

features_A = sequence_lengths(Population(A)).aggregate()
features_B = sequence_lengths(Population(B)).aggregate()

vals_A, counts_A = features_A["sequence lengths"]
vals_B, counts_B = features_B["sequence lengths"]

mean_A = (vals_A * counts_A).sum() / counts_A.sum()
mean_B = (vals_B * counts_B).sum() / counts_B.sum()

print("Expected number of actvities per sequence:")
print(f"  Population A: {mean_A:.2f}")
print(f"  Population B: {mean_B:.2f}")
# Expected number of actvities per sequence:
#   Population A: 4.14
#   Population B: 3.67
```


In practice, `acteval` represents distributions as weighted histograms — `(values, weights)` tuples — and computes EMD via the [POT](https://pythonot.github.io/) library.

---

Note that we are measuring the distance between populations of schedules. Compared population don't need to be comprised of the same persons or be the same size. Comparison is not therefore pairwise.


## Bringing it all togther

The `sequence_length` feature is a useful comparison, but obviously there's a lot more going on in activity schedules than their lengths. The `acteval` strategy is to simply consider and measure the difference between loads and loads of features.


For example, we consider the number of times each activity type occurs (`participation rates`), the number of times each transition from one activity type to another occurs (`2-grams`), the durations of each activity type (`durations`), and so on. The full catelogue, and which of these are used in the default evaluation configuration are available from `acteval.features.catalogue`:

```python
from acteval.features import catalogue

print(catalogue.list_features().to_markdown())
```

|    | domain         | group                    | config_key         | description                                                                            | in_default_config   |
|---:|:---------------|:------------------------|:-------------------|:---------------------------------------------------------------------------------------|:--------------------|
|  0 | participations | sequence lengths        | lengths            | Distribution of number of episodes per person.                                         | True                |
|  1 | participations | participation rate      | rates              | How many times each person participates in each activity.                              | True                |
|  2 | participations | pair participation rate | pair_rates         | Co-participation counts for all activity pairs.                                        | True                |
|  3 | participations | seq participation rate  | seq_rates          | Participation rates keyed by sequence position (e.g. '0home', '1work').                | False               |
|  4 | participations | enum participation rate | enum_rates         | Participation rates keyed by n-th occurrence of each activity (e.g. 'home0', 'home1'). | False               |
|  5 | timing         | start times             | start_times        | Start-time distribution per activity × occurrence index.                               | True                |
|  6 | timing         | durations               | durations          | Duration distribution per activity × occurrence index.                                 | True                |
|  7 | timing         | start-durations         | start_durations    | Joint (start, duration) 2-D distribution per activity.                                 | True                |
|  8 | timing         | joint-durations         | joint_durations    | Joint (duration_i, duration_{i+1}) distribution for consecutive activity pairs.        | True                |
|  9 | timing         | start times by act      | start_times_by_act | Start-time distribution per activity (no occurrence index).                            | False               |
| 10 | timing         | end times by act        | end_times_by_act   | End-time distribution per activity (no occurrence index).                              | False               |
| 11 | timing         | durations by act        | durations_by_act   | Duration distribution per activity (no occurrence index).                              | False               |
| 12 | timing         | time consistency        | time_consistency   | Per-person flags: starts at 0, ends at 1440, total duration equals 1440.               | False               |
| 13 | transitions    | 2-gram                  | 2-gram             | Consecutive activity pair (bigram) counts per person.                                  | True                |
| 14 | transitions    | 3-gram                  | 3-gram             | Consecutive activity triple (trigram) counts per person.                               | True                |
| 15 | transitions    | 4-gram                  | 4-gram             | Consecutive activity quad (4-gram) counts per person.                                  | True                |
| 16 | transitions    | full sequences          | full_sequences     | Per-person indicator for each unique full abbreviated tour string (e.g. 'h>w>h').      | False               |

...Which is a lot, so to be more useful for quick comparisons, we support aggregations to **group** and **domain** levels. The highest level, **domain**, consists of the following:

- **participations**: the taking part in activities
- **transitions**: the moving between activities
- **timing**: when and for how long activities occur

Domain-level results provide high level evaluation of populations and expose key trade-offs. Perhaps one model is better at participations, and the other at timing, for example. Additionally, the domains have somewhat different units (participation rates, transition reates and time (in days)) so we don't aggrgeate further, although you certainly can.

The `acteval.Evaluator` orchestrates all these comparisons, against a target population of schedules, in an efficient way. It also looks after descriptive metrics and non-density estimation features, such as for measuring feasibility and creativity. More on these in the future.

```python
from acteval import Evaluator

A = urban_workers(1000)
B = leisure_dominant(1000)
C = education_leaning(1000)

evaluator = Evaluator(target=A)
print(evaluator.compare({"B": B, "C": C}))

# EvalResult — 2 model(s): B, C
#                        B         C
# domain                            
# creativity      0.006521  0.016128
# feasibility     0.000000  0.000000
# participations  0.263421  0.205167
# timing          0.061743  0.031365
# transitions     0.243480  0.184492

```

## More to come

- Creativity and diversity
- Attributes
- Reporting
- Pair-wise

