---
layout: post
title: Evaluation
date: 2025-07-13
description: measuring success
tags: ML, evaluation
categories: Dev
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

Two years into PhD. How to measure the quiet trickle of progress?

I've been carefully writing and rewriting a paper. Sometimes in a furious state. But now, more or less past the finish line, I have the chance to look back and write some of the stuff here that got lost in the process.

My research is just old ideas. But it's new ideas for an old problem, and as such, how to play the game is slightly in flux. The following will be about how to play the game of measuring success, i.e. of *evaluation*.

# Primer

The mention of *evaluation* can be followed by an immediate mental slide into some metric or other, a distance, a probability, or some other quantity. But what we measure matters more than how we measure it.

We evaluate our model/device/thing to convince ourselves, plus some reviewers and all other interested parties, that our thing is good. Specifically, that it is good, or fit, for some specific purpose or application. Maybe it predicts the future. Maybe it can tell a dog from a cat. Or maybe it's for zesting lemons. The important point, obviously but easily forgotten, is that our evaluation needs to demonstrate fitness for that purpose. It should simulate the real intended use of our model/device/thing as closely as possible.

# Evaluation in Machine Learning

***For the ML congregation, the train, validate, test divide is sacrosanct, and benchmarking tasks are plentiful.***

Some research whittles away at evaluation benchmarks so entrenched and emblematic of progress that tiny fractions of progress grant canonisation. These highly reproducible evaluation frameworks are at least partially responsible for the rapid ascendancy of ML.

There are, however, some wrinkles. With attention and careers on the line, there is the temptation to misbehave, and even without any bad apples, test data manages to creep into train or validation sets. Big models have also paved the way for progress by grad-descent, where state-of-the-art model designs, specifications, and seeds are achieved using a faithful flock of grad students.

So you follow the lore, you split your data and evaluate. But still, people look over your shoulder and grumble. The split is supposed to simulate the real application; random splits may not suffice. If you predict the future, then testing on random slices from the past is unrealistic. If your model is for a new place, at a new time, or in new conditions, your test data should simulate as such.

Most recently, model capabilities have ascended past human experts and saturated the classic benchmarks. We now train models to undertake more complex and difficult tasks. It's increasingly not so obvious or easy to evaluate such requirements. Some complex tasks, like mathematical proofs or coding, are intrinsically verifiable, but others are not. How to evaluate whether customers are happy with your bot or if your Ghibli-style self-portrait is any good?

## Evaluation Elsewhere

***Before ML, there was the Age of Statistics.***

The statisticians of the time, without much data or compute, built models upon structures of knowns, in which the capacity to learn was carefully rationed and controlled. Parameters were few, unknown ones especially, and so capacity was limited and tasks simplified. Without much data to go around, nothing was withheld and the evaluation of a model was more an evaluation of the modeller, about what they knew or could do, rather than what the model could do. But this wasn't a problem, because back then you could look at a model, see it all, and more or less understand it and know that it was good.

Having a little model that you can truly understand is reasonably sufficient grounds for good evaluation. But little models can't do difficult tasks, and so we sometimes stack them up. Human behaviour is a nice example. Pick a simple choice - *should I travel to work by car or by bike?* - find some data, specify a little model, train it, maybe test it. Looks fine. Add some complexity - *should I also go shopping today?* No problem. Stick another model on. But then, *what time should I leave the house? Isn't tonight bowling night?* and so on. Quickly, we have a very complicated collection of simple models, which we don't truly understand, and we're not sure how well we truly evaluated it/them.

## Generative Modelling

***Enter generative AI, the amalgamation of probabilistic modelling and deep machine learning.***

A potent combination of uncertainty and opacity. In its purest case, a generative model aims to learn the distribution of some observed data sample. This data and distribution might be images of cats, or 18th-century poetry, or for me, people and their choices. But we'll just call the distributions $$P(X)$$.

Once you've learnt $$P(X)$$, you can generate new samples from it. As well as access to a near-infinite diversity of cat images, generating new samples allows us to surrogate processes that are either too complex or unknown to be otherwise simulated. Like a chat with a human.

Generative modellers are also working with big, complex distributions. The possible space of all possible cat images, for example, is massive. Consider an image size of N pixels, each pixel has three channels (RGB) with 256 possible values. This forms a joint distribution of pixel channel values, with $$256^{3*N}$$ possible images.

## Generative Evaluation

When we tackle a purely generative problem, we have *some* data from $$X$$ and use this to model $$P(X)$$. Skipping forward a bit, we've finished training our model and we want to evaluate it. But where is our test split? Can we even make one? Consider the beloved (my words) Variation Auto-Encoder (VAE) that (after some fiddling) provides a mapping between a random latent prior $$P(Z) ~ N(0,1)$$ and our desired distribution $$ P(X) $$, such that when we want a new cat image, poem or person, we randomly sample a $$z$$, pass it through the model, and out pops a brand new cat image, poem or person. We can test some mechanisms of the model, but ultimately, the evaluated case needs to capture this generative process, which is always out of scope of the training data, because its input is a random sample from $$Z$$.

---
**NOTE**

Most generative models actually use some form of conditionality. The big boys, image and text generation models, are typically text prompted, for example. But usually there is at least some generative or probabilistic process remaining. For example, they learn $$ P(X|Y) $$.

---

***New recipe then. Density estimation.***

Step one. We claim we have a dataset worthy of all real cat images. Nobody will say otherwise; the data is fine. Step two. We measure out a large number of generated cat images from our model. Step three. Mash the real cat images into a probability distribution. Step four. Mash the synthetic images into a probability distribution. Final step. Do they look similar? Yes? Great! You have made a delicious generative model. No? Relax! Try re-mashing harder.

The sticking point is that estimating probability distributions from a finite number of samples, sprinkled across a very large, complex and unknown distribution, is somewhere between really difficult, infeasible and impossible. Upon minor reflection, this was in fact the whole point of generative modelling. We want to fill in those gaps in the data, because we don't know them. But because we don't know them, we are now going to struggle to evaluate the quality of our filling.

We could withhold a sprinkling of real data as a test set, but it won't be useful in the great expanse of possible $$X$$ cat images.

## Quality vs Diversity

Nobody is perfect. But if we were perfect, at density estimation, then we would expect our generated samples to be both realistic individually and in aggregate. For example, we would expect our cat images to individually look like cats, and collectively, to span the full diversity of possible cats.

But we are not perfect and wouldn't even know if we were because we don't know what perfect looks like. So, as we fail, it is useful to know how we failed. Did we drift away from individual quality or from collective diversity? A common indicator of poor image quality was blurryness, and then as things got better/bigger, too many fingers. Over the same time period, people became more chill about diversity. Yes, we would like some diversity. We certainly don't want to collapse into no diversity. But do we need to see everything?

It is probably easier to evaluate individual quality in more cases. You can just have another model count fingers or even pay a human to give the thumbs up or down. Plus, individual quality is more of a priority in many applications. Why train a bot to speak the truth when you can just have it be agreeable? Just please don't be Mecha Hitler. As such, there has been a shift towards evaluating quality over diversity.

# Final Thoughts

But generative diversity really matters sometimes. My work generates human choices, specifically the actions we take and when, over the course of a day. These vary both from person to person and for the same person, from day to day. Sometimes I go for drinks after work, sometimes I sit at home all day watching brain rot. To usefully and fairly model the real world, I want to model this diversity and I want to model it correctly. 
