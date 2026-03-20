---
layout: post
title: The unreasonable difficulty of modelling reasonable people
date: 2026-03-03
description: parallels between modelling language and modelling people
tags: ML, LLM
categories: Dev
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

*What can language models teach us about modelling humans? And what should we be careful not to learn?*

---

# A brief history of making things up

Forty years ago, language modelling was pretty theoretical. Linguists and statisticians, working with limited data and hard-won assumptions about the structure of language, strived and mostly failed to generate realistic language (certainly compared to the state of the art today). Models were tidy, interpretable, and [quite bad at language](https://sites.google.com/view/elizaarchaeology/try-eliza).

Modern data-driven approaches have *replaced* these old approaches. Transport modelling, and human behavioural modelling more generally, finds itself roughly where language modelling was forty years ago.

# The transport modeller's predicament

***We have a lot of structure and not enough data.***

Travel demand models - discrete choice, activity-based, and so on - are carefully specified. We define utility functions from the literature or based on available data, then estimate or assume the parameters. Models encode decades of behavioural theory. You can look at them and, if you are an expert, sort of understand them.

Early in my tranport modelling career I went to Singapore to sales pitch MATSim. A hard sell but my audience was very polite and let me finish. It turned out that their only concerns were time and cost. Speciifcally, could we build it faster than a four-stage model? I asked *how much for a four-stage?* - 18 months aparently, and $,$$$,$$$ squid.

Despite this cost, such transport models are somewhat limited compared to language models. Despite taking years and vast sums of money to build, they only work for a single place and time. Increasingly they are also being pushed to model more complex scenarios in more precise ways, beyond the sensible boundaries of existing theory and practical implementation.

So why not follow language modelling's lead and throw data at the problem? The easy answer is that we don't really have the data. Language models were trained on essentially the entire written output of humanity — the GPT-3 corpus alone was something like 45 terabytes ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). The equivalent for travel behaviour would be a continuous, high-resolution record of where everyone goes, why, and with whom. We have travel diaries,and increasingly, some mobile phone data. But it is patchy, expensive, and often flawed or insufficient.

So transport modellers occupy an awkward middle ground. Not enough data to go fully empirical. Too much complexity to stay fully theoretical.

# An opportunity

Thanks to an apparent disregard for intellectual property, trademarks, and even personal privacy - perhaps irresponsibility - language models represent the outer limits of what is possible. But more efficient, safe, and sensible data-driven approaches certainly exist (see my previous love letter to [VAEs](https://fredshone.github.io/blog/2025/VAEs/)).

LLMs have also shown us the value of scale. They generalise to new situations using context, without retraining, so don't need to be constantly retrained. They achieve this in part through [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) — shifting knowledge learned in one domain to another.

Perhaps we can then imagine a transport model that works anywhere, at any time, and for any scenario required. We could have a model that predicts human choices, whether in a rural town in the US or a European city, in the near or far future, for scenarios ranging across domains and applications.

# The fundamental problem

***Should a transport model hallucinate?***

Modern LLMs have arguably already moved beyond language and increasingly imitate human behaviours. By training on our data they have learnt to imitate our ethics, our choices, and even our beliefs. But they also can't be trusted to (i) be correct, and (ii) more subtly, to represent reality faithfully.

I'm going to argue that being incorrect — or hallucinating, if you prefer — is not such a big deal. Simply put, transport models are wrong anyway. Predicting the future is just too hard. Not that I think we can't do better.

The latter problem, however — representing the diverse and often unexpected range of human decision-making — matters. LLMs are increasingly being shown to [homogenise](https://arxiv.org/html/2504.05228v1) towards the most common outputs - mode collapse on a grand scale, perhaps. Not everyone should buy a car, get an office job, and work nine to five in our models. Someone has to buy a bike, open a bookshop, and enjoy early morning walks. Someone also has to work two night shifts, pick up the kids at mid-day because they have a fever, and unwisely buy a motorbike. Diversity is important - because it's real.

# The RL problem

***Reinforcement learning broke language models open. It will probably not be so useful for us.***

The leap from "impressive autocomplete" to something disturbingly competent is substantially the product of [reinforcement learning from human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022): train the model to produce outputs humans prefer, penalise the bad ones, let it optimise. This works because there exists a coherent notion of a *good* answer.

Now try to apply this to human behaviour modelling. We have utility theory to maximise — this gets us about as far as predicting whether a person will take the bus. But what about when to get groceries? Pick up a coffee? See a friend? As we push models further, there is no clean objective function.

Even when we can specify one, like *get to work on-time!*, we would not want to optimise for it. The whole point is to predict what real, flawed, inconsistent humans will actually do — not what perfectly rational agents would do. We want the person who drives to the gym and takes the lift. We want the commuter who takes the slow train on a sunny day. Super-human performance is the goal for language models. For human behaviour models, it would be a catastrophic failure mode.


