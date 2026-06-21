---
layout: post
title: Beneath the hype of ML - an activity schedule modelling case study
description: >
  When and how should we use AI for human behaviour modelling?
date: 2026-06-21
tags: AI, human behaviour modelling,
categories: Dev
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

This post accompanies a presentation for 2026 Modelling World, in the Activity Based Models session, called "Beneath the hype - actual advantages of ML - an activity schedule modelling case study".
 
## Motivation
 
My motivation for this talk is to carve out a clear case for when and how it makes sense to think about using ML, especially deep or "black box" ML, for human behavioural modelling. I also hope to introduce a few new useful concepts, and seed a little doubt or even discontent in orthodox approaches.
 
## A primer
 
A quick thought experiment to mull over.
 
I have recently become a father. Understanding and predicting my sons various behaviours - when he sleeps, when he will eat and so on, is hard. I have to conclude that he is not a rational utility maximising decision maker. At best he copies. But more often his actions are seemingly random.
 
Consider then a robot in an amazon warehouse. This machine rationaly navigates on a perfectly optimised trajectory, moving parcels around a maze of shelving and conveyors.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/fig1.png" caption="Rationality and optimisation in decision makers" max-width="90%" %}
</div>
 
Then consider yourself, and the decisions you make as you go about your day, when to leave the house, if to go shopping and so on. Presumably your behaviour falls somewhere between my son's and the robot, between chaotic and inexplicable, and logical and explainable. But which is it closer to? The mysterious infant or the logical robot?
 
## A baseline
 
First consider the workhorse of human behavioural modelling, the discrete choice model.
 
In the human behavioural modelling domain, the logistic regression, and it's relatives, have dominated the last 70 years of both theory and practice.
 
The case for switching such models to deep ML is primarilly about capacity and complexity. Capacity to learn more complex patterns, in particular non-linearities without having to manually create and test all such possible features. It is also about speed. ML is fast, not just to run, but to train, to calibrate, to find data for, to implement, to upskill and recruit even. Without the burden of behavioural theory it is possible to build and use much faster, more flexibly and cheaply.
 
But of course this comes at some cost. A common critique is that ML is not interpretable and so can't be trusted. But this critique slightly misses the mark or is imprecise. Easy approaches to explaining how ML models work exist. When a model is fast it is pretty trivial to simulate elasticities and the like. Instead the critique should be that without interpretability, specifically without a theoretical structure for model parameters, the modeller has no control over the model's behaviour.
 
Given a logistic regression, for example, the modeller can simulate the novel response of a person to some change in preference or sensitivity. This allows the modelling of scenarios or behaviours not in the data. Essentially models with behavioural theory foundations allow the modeller to extrapolate or model beyond the data, whereas a more ML approach can only operate within data.
 
Being stuck in the data might sometimes be ok. Firstly there is a lot more data nowadays but also giving the modeller more control is a double edged sword. Will the modeller use this power for good? To carefully simulate a likely future scenario perhaps. Or will they use it for bad? To spuriously fit a baseline or achieve a BCR?
 
Overall I would say that, in this context, the case for ML is weak. That if a modeller knows what they are doing, the gain in speed is not worth the loss of control.
 
## A more useful case study
 
In the context of modern transport demand models, discrete choice models are increasingly just components. Little cogs in the grand machine that is an activity-based model.
 
Consider human activity scheduling - the choice of what activities to participate in, when and for how long. Scheduling is foundational to the motivation of activity-based models - to treating time as a resource. Accordingly all activity-based models engage in some sort of approach to scheduling.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/fig2.png" caption="Activity scheduling" max-width="90%" %}
</div>
 
Real world activity scheduling is complex. Schedules are the result of a combination of physical constraints like time and space, preferences such as if to work late or visit a friend, and interactions with the world, such as sharing a vehicle or meeting congestion.

## The classic compositional approach

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/compositional.png" caption="Compositional model" max-width="75%" %}
</div>
 
Current approaches to scheduling are complex. Approaches within activity-based models also vary, there are basically as many approaches to scheduling as there are activity-based models.
 
I have first hand experience, from building my [own](https://github.com/fredshone/composhed). It strings together multiple model components, some are discrete choices, such as daily activity patterns or tour and sub-tour types, others are continuous variable choices, such as durations. All these choices are then assembled together into cohesive schedules using some logic or other.
 
The primary criticism I want to make is that these models are slow. Slow to run, slow to estimate, slow to calibrate, slow to develop and generally hard to use. It takes years and $$$$$$$ to develop new activity-based demand models and much of this delay comes from the need for such complex arrangement of sub-models, for scheduling and other logic.
 
Speed/cost: D
 
The complexity of such models also challenges the explainability and controllability advantages of traditional approaches. Interactions between models and with the algorithms used to assemble choices into valid schedules are challenging to intrinsically understand and predict. But provided the modeller knows what they are doing and is prepared to put in some time, these models are somewhat controllable.
 
Controllability: B
 
Again, given sufficient time these models will do a pretty good job at predicting schedules, certainly the most common schedules. However, the sub-model specifications and their interactions are heuristic decisions and ultimately create simplifications that limit the realism of modelled schedules.
 
Realism: B-
 
 
## ML for flexibility and simplicity
 
Much is made of the learning capacity of ML - it's ability to form and use complex features. But this capacity extends to a capacity for complexity of model inputs and outputs. Specifically, ML enables architectures that can injest and excrete complex structured data, including human activity schedules.
 
I built a model to do this called ActVAE. It lives [here](https://github.com/big-ucl/caveat). It's a deep generative model, specifically a Variational Auto-Encoder with a Conditional Prior. But this isn't immediately important.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/actvae.png" caption="ActVAE model" max-width="75%" %}
</div>
 
This is fast because we can train a single model to replace the complex compositions of sub-models. This model can be calibrated and trained quickly end-to-end, and can generate schedules 100s of times faster. It also turns out that the schedules, specifically the distributions of schedules, that these models output are as or more realistic. This is primarily because people are complex and varied and so are their schedules. Released from the heuristics and simplifications of sub-models, deep ML approaches can capture this complexity and variance better.
 
But this all comes with a severe loss of controllability. The model is a black box. If it doesn't do what you want, you have to deal with this outside the model.
 
Speed/cost: A
Realism: B+
Controllability: F
 
## A logical extreme
 
We can take the pro-ML ethos to it's logical extreme and spin up an LLM. This has the advantage of leveraging a great deal of observed human behaviour. A lot of this observed behaviour is tangential, people arguing on Reddit for example, but some of it will be relevant. We can also fine tune them for our task and benefit from their ability to understand and follow instructions.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/actllm.png" caption="ActLLM model" max-width="75%" %}
</div>
 
I have code for testing such models [here](https://github.com/fredshone/actllm).
 
Bad news is that this is slow and costly and inefficient and that realism plummets. Good news is that the models are very easily controllable using natural language instructions. Annoying statements like "You are an expert transport planner" appear to actually work. But you can also give LLMs more concrete instructions, like; "make sure all schedules start and end at home".
 
Speed/cost: F
Realism: F
Controllability: B

## Receipts

These scores are not speculation. I have done the experiments. ActVAE is clearly fast and LLMs are slow. ActVAE surprisingly does rather well at density estimation (measuring this is a whole other [thing](https://fredshone.github.io/blog/2026/acteval-distances/)), and only lags marginally behind the other models in controllability.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/tab1.png" caption="Evaluation" max-width="90%" %}
</div>

## Why does the ML approach do realism well?

ActVAE does well at realsim, at least in part, because it is an explicit generative model. Which means it explicitly learns random variations in scheduling data. This is important because random variations dominate explanatory variables.
 
ActVAE is a latent generative model which learns a compressed (latent) representation of all training data. This compressed representation is composed of two parts, the first is the explanatory variables, age, income and so on, the second, a random distribution. If we measure the "usefullness" of the two components (using mutual information estimation), we find that the random component is almost 10 times as useful as the explanatory variables. Basically, when modelling schedules, random variation dominates.
 
This should not be surprising. Humans are complex and schedules are subject to all kinds of interactions and perturbations that we can't or don't observe in our data. I call this the "alarm clock problem". Where an individual's preference to get to work early is hidden almost entirely by if they remember to set their alarm. It could also be the "where are my keys?" theorem or "there's a leaf on the track" dilemma.
 
Getting this random variation looking correct is a secondary concern of traditional composed approaches. It is an outcome of the interactions of error terms across all the submodels. In contrast, for a generative ML approach, it is a first class citizen.
 
## A framework for thinking about realism and the failure of LLMs
 
Ultimately we want models that can output correct distributions of schedules. But we can decompose this distribution into intra and inter-schedule variation.
 
Intra-schedule variation considers the construction of individual schedules and how realistic they are. In the image generation domain, this is commonly referred to as sample quality and is sometimes as simple as checking that images of hands have the correct number of fingers or that shadows are correctly placed. For schedules this might be some structural requirement, like all schedules start and end at home.

<div class="text-center">
{% include figure.liquid path="assets/img/mw2026/fig3.png" caption="Inter-intra variation" max-width="90%" %}
</div>
 
Inter-schedule distribution includes the aggregate distribution of activity participations and start times across the whole population. When do people typically travel and why for example.
 
For human behaviour models both intra and inter-schedule distribution is key.
 
So why did the LLM fail at realism? Well, amongst many other reasons, it just isn't trained or designed to be good at inter-sample distributions. This isn't a novel finding, ask a model 50 times to name a US state and it will not name all 50 states. It will [likely](https://arxiv.org/html/2510.01171v1) keep saying California or maybe Texas. But it is a pretty catastrophic finding if you are trying to use an LLM for human behavioural modelling.
 
## Conclusions
 
When to use ML?
- you are modelling a process with complex inputs or outputs
- you want faster or cheaper
- you care less about controllability, for example for modelling system shocks, like pandemics, and more about representing diversity

Where you have a noisy process you should be using a generative approach.
 
Rather than thinking of replacing whole frameworks with ML, think of swapping components in where it makes sense for that specific use-case.
 
Don't trust LLMs to behave like humans.
 
## Final thought
 
Back to our original question. Are people noisy baby-like choice makers? Or are they careful logical robotic optimisers?
 
I argue more for the former. Although in the long term (think billions of years) we are definitely optimised and have gained the capacity to optimise. These hard-bred behaviours did not have navigating the complexities of the London Underground in scope. Falling back on a black-box approach to behaviour modelling is perhaps lazy but also maybe more realistic and in complex cases, like activity-based modelling, pragmatic.
