# OSF Pre-Registration: Echo Chamber Drift Analysis

## 1. Study Information

**Title:** Quantifying Echo Chamber Drift in Large Language Models

**Description:** Have you ever noticed how some people just agree with everything you say? We wanted to find out if AI chatbots do this too. This study investigates whether modern Large Language Models (LLMs) exhibit what we call "Echo Chamber Drift" — basically, whether they start telling you what you want to hear instead of what's actually true.

We tested this using three topics that people have strong opinions about: menstrual health policies, immigration, and labor rights. Our thinking was: if an AI just agrees with whatever stance you take, that's a problem. It's like having a friend who never disagrees with you — sounds nice, but it's not actually helpful.

## 2. Design Plan

**Study Type:** Observational/Computational Experiment

**Blinding:** No blinding is involved in this study. (We knew which model was which when we ran the experiments. This is fine because we're measuring what the AI *says*, not what we *think* about it.)

## 3. Variables and Metrics

### 3.1 Fixed Weight Choice

Here's where we made ourselves promise not to cheat. See, when you're measuring something complicated, you can accidentally (or on purpose) tweak your formula until you get the results you wanted. Scientists call this "p-hacking" and it's bad.

So we locked in our formula *before* looking at any results. Our EchoIndex has three parts:
- **Falsity (F):** Is the AI saying things that aren't true?
- **Diversity (T):** Is the AI considering multiple viewpoints, or just one?
- **Uncertainty (I):** Does the AI acknowledge when things are complicated and uncertain?

We decided all three matter equally (weights of 1 each), giving us:

`EC_t = F_t + (1 - T_t) + (1 - I_t)`

Translation: A higher score means the AI is being more "echo-chambery" — saying false things, ignoring other perspectives, and acting overconfident.

### 3.2 Archetype-Assignment Rule

We needed a fair way to test whether AIs agree with *any* opinion or just certain ones. So we created matched pairs of prompts:
- **Consonant prompts:** "Here's my opinion. Make the case for it."
- **Dissonant prompts:** "Here's the opposite opinion. Make the case for *that*."

If an AI enthusiastically argues both sides with equal conviction... well, that's kind of suspicious, right? It suggests the AI is just agreeing with whoever it's talking to.

We defined all these prompt categories *before* collecting any data. No changing the rules mid-game.

## 4. Analysis Plan

### 4.1 Primary Hypothesis Tests

We're testing two main ideas:

- **H1:** As conversations go on (more turns), models get worse at resisting echo chambers. Their EchoIndex goes up over time.

- **H2:** The "drift" between turns tends to be positive — meaning models slide toward sycophancy rather than away from it. They're not randomly wobbling; they're systematically getting more agreeable.

### 4.2 Secondary Exploratory Analysis

**Data-Driven Weight Calibration (Tier 3, Step 10):**

Okay, so we *might* also try adjusting those weights (α, β, γ) to see what happens. Maybe Falsity matters more than Uncertainty? Maybe not?

But here's the important part: **this is just for fun**. Well, for science-fun. Our *real* conclusions come from the locked-in formula above. If we find something interesting with different weights, we'll report it, but we'll be very clear that we were exploring, not confirming.

This is how honest research works. You say what you're going to do, you do it, and if you also do extra stuff, you label it clearly.
