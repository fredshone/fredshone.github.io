---
layout: post
title: Human articicial intelligence
date: 2026-03-19
description: Tangential: superhuman AI and why it won't help us understand people
tags: ML, LLM
categories: Dev
giscus_comments: false
related_posts: false
toc:
    sidebar: left
---

*What to make of the drive for superhuman intelligence regarding the problem of modelling humans?*

---

This post is to simply make the point that the hunt for super-general-intelligence (SGI) is tangential to what we need for modelling humans and probably systems of humans.

# What is intelligence?

Psychologists, biologists, and computer scientists have been arguing about the definition of intelligence for some time.

My first, unastonishing take, is that checking the length of a tape-measure with the same tape-measure is probematic. Humans defining a thing, using that thing is self-referential or circular. 

You can escape this loop with an evolutionary inspired take on intelligence. Broadly, intelligence isn't an abstract universal property. Instead it's fitness-for-purpose for context.

Therefore intelligence changes when the environment changes.

# Hooman intelligence?

AI has caught up with me in some ways. It has been better at me at Go for a while and now probaby writes better code than me. I expect it will be a better researcher than me given some appropriate goals.

Surpassing humans isn't always a high bar. Humans are genuinely bad at stuff. We're poor at reasoning under uncertainty. We misread probabilities. We discount the future too steeply. We're overconfident in familiar domains and panicked in unfamiliar ones. I'd argue that we're a little over-optimised for arrising from the primordial soup. But suprisingly bad at navigating sensibly across London.

# Super intelligence and becoming inhuman

Reinforcement learning (RL) is at the core of pushing past huan capability. Rather than being anchored in human observations, it can push past into uncharted challenges or context. The mechanism is elegant: give the system a reward signal, let it explore, and iterate until it finds strategies no human designer would have thought of.

RL works best in environments with clear reward signals, well-defined rules, and computable feedback. Games. Simulations. Formal problems with ground truths.

These are precisely the environments where humans are weakest — combinatorial, high-dimensional, fully observable. Pushing AI capability in these domains is tangential to modeling humans.

# How to be human

The human workday doesn't have a clean reward function. We use heuristics, habits, identity, shortcuts, and instinct. Our "objectives" operate on timescales from seconds to decades simultaneously, and they conflict with each other constantly. The reasons we give for our choices are often retrospective. We often can't be trusted to knwo why we did something at all[1]. 

This is a tricky reward (or utility[2]) function to devise, and certainly not what sits at the heart of LLMs and other models on their way to SGI[3]. Instead I suggest we simply imitate the pattern. Absorb the mess and reflect it back without needing to reduce it to a clean objective[4].

___

[1] See [choice blindness](https://pmc.ncbi.nlm.nih.gov/articles/PMC4172758/)

[2] There are some parallels between RL and classic behavioural based choice models. But they are out of scope here. Simply we can say that utility based approaches are also optimisers and that this is also problematic.

[3] We could still perhaps spin of our own SGI (Sh*t General Intelligence)

[4] This argumant is heavilly reliant on the assumption that we model humans as agent that interact to simulate collective behaviours. Rather than simply cut the corner and model the collective sytem directly, for example. Such a system would certainy need to be superhuman and I don't have much time for it because (i) such complexity seems intractible, and (ii) how to trust a system that one-shots the future without showing us any working?

