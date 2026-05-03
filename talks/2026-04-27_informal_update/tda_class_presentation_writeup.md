# TDA Class Presentation Writeup

This writeup is meant to complement [`docs/experiment_log.md`](../../docs/experiment_log.md), not replace it.
The experiment log already contains the detailed run history and result tables.
This document focuses on the parts you need to explain the project clearly in a class presentation:

- the problem setup
- the research questions
- the TDA intuition
- why the method was designed the way it was
- what the main findings mean
- what to claim carefully
- what questions people are likely to ask

## Short Version

If you need a two-sentence summary:

> I study whether topological summaries of speech representations can detect spoofed or deepfaked audio in a way that stays interpretable across datasets. The main result so far is that low-frequency cubical persistent-homology structure is the most robust signal, and a staged topology-only neural classifier improves over a linear topology baseline while still preserving that core structural story.

## What Problem This Project Is Solving

The broad problem is audio deepfake detection: given a speech clip, predict whether it is bona fide human speech or spoofed / synthetic speech.

A lot of strong modern systems attack this with end-to-end deep learning on spectrograms or waveforms.
Those systems can work well, but they are usually hard to interpret.
If they generalize poorly across datasets, it is often unclear what exactly failed.

This project asks a narrower but more interpretable question:

> Can topological structure in speech-derived representations distinguish real from fake audio, and if so, what kind of structure is most robust?

That framing matters because the goal is not only classification accuracy.
The goal is also to learn something structural about the signal.

## Core Research Questions

These are the questions worth stating explicitly in your talk:

1. Can persistent-homology-based features detect spoofed audio at all?
2. Which representation works best: point-cloud topology or cubical topology on spectrogram fields?
3. Within the cubical pipeline, what parts of the spectrogram matter most?
4. Which homology dimensions matter most?
5. Does the same structural story survive across datasets, or is it only fitting one benchmark?
6. If we keep the representation topological but use a more expressive classifier, do we gain anything without losing interpretability?

That sequence gives the presentation a clean arc:

- first show that topology is viable
- then show that the cubical route became the strongest branch
- then show that the low band and `H1` are the central motifs
- then show how stable that story is across datasets
- then show the neural follow-up as a controlled extension

## What Makes This A TDA Project Instead of “Just Another Audio Classifier”

The distinguishing feature is that the model does not classify directly from raw audio or spectrogram pixels.
Instead, it builds topological summaries and classifies those summaries.

That means the project is centered on questions like:

- what connected components survive?
- what loops survive?
- which structures are stable under perturbation?

In other words, the project is using topology as the representation language, not just as a decorative add-on.

That is why the ablation story matters so much here.
If low-band `H1` consistently survives as the useful signal, that is a structural claim about the data representation, not just a model-selection artifact.

## Why Persistent Homology Makes Sense Here

You do not need a heavy formal explanation in the talk.
What you need is a useful intuition.

Suggested explanation:

> Speech is not just a bag of frequencies. It has shape: repeated patterns, continuities, transitions, and local structures that evolve over time. Persistent homology is a way to summarize which geometric or field structures persist across scales, while suppressing structures that look like noise.

For this project, persistence is attractive for three reasons:

1. It is scale-based.
   Fake audio may differ from real audio not in one exact coordinate but in how structure appears across thresholds or neighborhoods.
2. It is robust.
   The project is explicitly interested in stable structure, not fragile pointwise cues.
3. It is interpretable at the block level.
   You can ask whether `H0`, `H1`, low-band structure, or broader field structure is carrying the signal.

## Why the Project Moved Toward Cubical PH

The project originally included multiple topological branches:

- Vietoris-Rips PH on feature point clouds
- weighted graph / flag-complex PH on point clouds
- cubical PH on mel spectrogram grids

The cubical branch became the strongest and most interpretable route.

Why:

1. A spectrogram is already a structured 2-D field.
   Cubical PH respects that grid structure directly instead of forcing everything into an unordered point cloud.
2. It is easier to localize what matters.
   Once the input is a field, you can do meaningful spatial/frequency ablations like keeping or dropping the low band.
3. It fits the research question better.
   The interesting question became “which parts of the spectrogram field carry robust topological signal?” and cubical PH is the natural language for that.

## Why the Low Band Became Central

One of the strongest empirical findings is that low-frequency spectrogram structure matters the most.

This is not just a random hyperparameter winner.
It became credible because the same pattern showed up repeatedly:

- keep-low variants improved over the full-field reference
- drop-low variants collapsed
- low-band `H1` stayed strong across settings
- the family continued to matter even when the exact winner shifted by dataset

A clean way to explain this:

> The low band appears to contain the most reliable topological differences between bona fide and spoofed speech. That does not mean the rest of the spectrum is useless, but it suggests the strongest stable cue is concentrated in lower-frequency structure.

Why might that happen?

- lower bands carry formant-envelope and voicing-related structure
- synthesis artifacts may be less convincing there in a topological sense
- higher bands may be more dataset-specific, noisier, or more brittle under domain shift

You should present those as plausible interpretations, not proven causal facts.

## How to Explain H0 vs H1 Without Overcomplicating It

You can explain the homology dimensions informally:

- `H0`: connected regions / components
- `H1`: loop-like or hole-like structure
- `H2`: higher-dimensional cavities

The project’s structural conclusion is:

- `H1` is the most robust component
- `H0` can help in-domain
- `H2` did not contribute anything useful in the current 2-D cubical setup

What that means conceptually:

> The useful difference between real and fake speech seems to lie less in simple connectedness alone and more in the more organized mesoscopic structure that `H1` captures.

Again, keep it qualitative.
Do not imply you have identified literal physical holes in speech.
What you have identified is that loop-level persistence summaries are more informative than component-only summaries in this representation.

## Why the Cross-Dataset Study Matters

This is one of the most important framing points in the talk.

If a configuration only works on one benchmark, it may just be overfitting that dataset’s quirks.
So the real scientific question is:

> Does the structural story transfer?

Your multi-dataset study is valuable because it goes beyond “best metric on one split.”
It asks whether the same topological motif survives:

- in-domain on 2019 LA
- transfer from 2019 LA to 2021 LA
- internal 2021 LA training
- bounded 2021 DF transfer

The strongest high-level conclusion is not that one single configuration wins everywhere.
It is this:

> The exact winner changes by domain, but the low-band cubical family survives across all of them.

That is scientifically more interesting than a brittle universal winner.

## Why the Neural Follow-Up Is Still Interpretable

A class audience may worry that the neural extension abandons the original TDA motivation.
Your answer is no, because the neural model is still topology-only.

Important distinction:

- not raw spectrogram neural net
- not audio+topology fusion
- not a large black-box architecture

Instead:

- input is still explicit topological feature blocks
- the blocks are semantically named
- you can still ablate blocks after training

That is why the staged MLP experiment is interesting.
It asks whether you can gain expressive power while preserving the structural story.

## Why the Staged MLP Matters

The staged model is not just “another neural net.”
It encodes a scientific hypothesis:

> Train on the most robust block first, then add weaker auxiliary blocks later, so the model does not learn brittle shortcuts too early.

In this project:

- Stage 1: low-band `H1` core
- Stage 2: add low-band `H0`
- Stage 3: add broader full-field topology

The important result is not only that staged MLP performed best among the topology-only neural models.
It is that its ablations show stronger dependence on the low-band `H1` core than the flat MLP or the linear baseline.

That means the staged procedure did not just improve metrics.
It improved alignment with the structural hypothesis.

## What the Main Story Is

If you need one clean narrative thread for the talk, use this:

1. Start with the general idea that spoofed audio may differ from real speech in stable geometric/topological structure.
2. Show that cubical PH on spectrogram fields became the strongest branch.
3. Show that low-frequency structure is the main recurring signal.
4. Show that `H1` is the most robust homology component.
5. Show that this story survives across datasets, even though the exact winner changes.
6. Show that a small staged topology-only neural model improves over a linear topology baseline while still relying primarily on the same robust core.

That is a stronger and more coherent talk than trying to present every experiment as equally important.

## What You Should Claim Carefully

There are several things you should be careful not to overclaim.

Do claim:

- topology is viable for this task
- the cubical branch became the strongest path in this project
- low-band structure is the most robust repeated finding
- `H1` is the most robust homology component
- the multi-dataset study suggests the structural principle survives domain shift better than some specific parameter choices do
- the staged topology-only MLP helps relative to a linear topology baseline

Do not claim:

- that topology is universally better than end-to-end audio models
- that the best number in one table proves a new state of the art
- that `H1` literally identifies one physical speech mechanism
- that the DF bounded transfer result is the final story for DF overall
- that the neural model has already beaten the strongest tuned classical cubical configuration

That last point is especially important.
The staged MLP beat the linear topology baseline, but it did not yet beat your strongest tuned classical transfer branch.
That is still a useful result.

## What Makes the Project Interesting to a TDA Audience

For a TDA class, the interesting part is not only “we got some decent metrics.”
It is that the project uses TDA to answer structured questions:

- which representation is topologically meaningful?
- which frequency region carries the most stable signal?
- which homology dimension matters?
- which patterns transfer across domains?

That is a better TDA story than a purely performance-driven benchmark talk.

## Suggested Presentation Structure

If you want a clean 8-10 minute flow:

1. Problem and motivation
   Explain audio deepfakes and why interpretability/generalization matter.
2. Research question
   Ask whether stable topological structure can separate real from fake audio.
3. Method intuition
   Spectrogram field -> cubical PH -> vectorization -> classifier.
4. Why topology
   Persistence as stable structure across scales.
5. Main empirical arc
   Cubical branch won, low band mattered most, `H1` was most robust.
6. Multi-dataset study
   Same family persists across 2019 LA, 2021 LA, and bounded DF.
7. Neural follow-up
   Small topology-only models; staged MLP best among neural variants.
8. Sample-level explanation
   Show one example where low-band/H1 fixes a wrong full-field decision.
9. Limitations
   Transfer is weaker than in-domain, DF is still bounded, no claim of SOTA.
10. Takeaways
   Low-band cubical topology is the main robust motif; staged topology learning is promising.

## Slide-by-Slide Speaking Guidance

### Slide: Motivation

Say:

> Most audio deepfake detection systems are strong but hard to interpret. I wanted to test whether topological summaries can recover discriminative structure in a way that is easier to probe and ablate.

### Slide: Method

Say:

> I treat the mel spectrogram as a 2-D field, compute cubical persistent homology on that field, vectorize the resulting persistence information, and then classify from those topological vectors.

### Slide: Why TDA

Say:

> Persistent homology is attractive here because it summarizes stable structure rather than isolated pixel or frame values. That makes it a natural tool for asking which aspects of the signal survive perturbation and which are probably noise.

### Slide: Main structural finding

Say:

> The biggest repeated finding was that low-frequency spectrogram structure dominates. When I keep the low band, performance improves; when I drop it, performance collapses.

### Slide: Homology finding

Say:

> `H1` was the most robust component across settings. `H0` can help in-domain, but `H1` is the cleaner transferable motif. `H2` did not help in the current 2-D cubical setup.

### Slide: Multi-dataset study

Say:

> The exact winning configuration changes by dataset, which is normal under domain shift. What matters is that the same low-band family keeps surviving, so the structural story is not restricted to one benchmark.

### Slide: Neural follow-up

Say:

> I then kept the representation purely topological and only changed the classifier. A staged MLP that learns low-band `H1` first and adds auxiliary blocks later beat both a linear topology baseline and a flat MLP.

### Slide: Example

Say:

> This sample is useful because the broad reference model scores it incorrectly, but the low-band and especially the `H1` branch recover the correct label. That makes the structural claim easier to understand concretely.

### Slide: Takeaways

Say:

> My main claim is not that one parameter setting wins every dataset. It is that low-band cubical topology is the most robust motif, `H1` matters the most, and topology-only neural models can extend the method without discarding interpretability.

## Likely Questions and Good Answers

### “Why not just use a standard CNN on the spectrogram?”

Good answer:

> That would probably be a strong baseline, but it would answer a different question. My goal here is not only performance; it is to understand whether stable topological structure carries signal and which parts of that structure matter.

### “Why does the low band matter so much?”

Good answer:

> Empirically it was the most robust region across ablations and datasets. My interpretation is that lower-frequency structure preserves more stable information about voicing and spectral-envelope behavior, while higher bands are more brittle or dataset-specific. That is an interpretation, not a final causal proof.

### “Why is H1 stronger than H0?”

Good answer:

> In this representation, loop-level persistence seems to capture a more discriminative intermediate-scale organization than component counts alone. H0 still helps in-domain, but H1 is the more stable transferable signal.

### “Did the neural model beat the classical model?”

Good answer:

> It beat the linear topology baseline and the flat topology MLP, which supports the staged-learning hypothesis. It did not yet beat the strongest tuned classical transfer configuration, so I treat it as promising rather than final.

### “How general are the DF results?”

Good answer:

> The DF results are bounded smoke/follow-up runs on a balanced subset from `part00`, so they show nontrivial transfer but are not the final word. They are useful evidence, not a definitive DF benchmark claim.

### “What is the scientific contribution?”

Good answer:

> The contribution is the structural story: cubical PH on spectrogram fields works, low-band structure is the most reliable motif, H1 is the most robust homology component, and staged topology-only learning can preserve that robust core better than flatter alternatives.

## What to Emphasize if Time Is Short

If you only have time for a short talk, emphasize these:

1. audio deepfake detection is the application
2. cubical persistent homology on mel spectrograms is the core method
3. low-band structure is the central finding
4. `H1` is the most robust homology signal
5. the low-band family survives across multiple datasets
6. staged topology-only neural learning is a promising controlled extension

Everything else is secondary.

## What to Leave Out if Time Is Short

If the talk is short, do not spend much time on:

- the early Vietoris-Rips branch details
- every hyperparameter sweep
- the exact train/dev/test counts unless needed
- the full DF caveat structure in detail
- every ablation number

Keep the attention on the structural conclusions, not the whole lab notebook.

## Terms to Define Briefly

These are worth defining in plain language:

- `spoofed audio`: synthetic or manipulated speech meant to imitate real speech
- `mel spectrogram`: a time-frequency representation of audio on a perceptually motivated frequency scale
- `cubical persistent homology`: persistent homology computed directly on a grid / image-like field
- `H0`: connected components
- `H1`: loop-like structure
- `ablation`: deliberately removing one part of the representation to see what mattered
- `transfer`: training on one dataset and evaluating on another

## One Good Closing Line

If you want a closing line that sounds like a research conclusion rather than a sales pitch:

> The main lesson from this project is that topology is not just usable here; it is diagnostically useful. It helped isolate a stable low-band structural motif that persists across datasets, and that is a more informative result than just finding one model that scores highest on one benchmark.
