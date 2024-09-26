# Thesis Outline

## Thesis Statement:

In terms of model behavior, there are some instances of hallucination that are identical to creativity. We investigate this in the following way:
Coding-Creativity model identification: linear probes, Mean differencing, Sparse Autoencoder (SAE)
Three different templates:

Factual: “Is there a python package for <task>”
Neutral: “Give me a python package for <task>”
Creative: “Make up a python package for <task>”

The idea is to identify a creative vector and a hallucination vector. Domain restriction (only python packages) should help identification
Do we need to build a causal model of this? Perhaps to better conceptualize our design?
Once we have a vector, we perform interventions, removing this vector from activations at each layer of the forward pass. 
Can we demonstrate that prompts like: “give me a python package for <task>” is more likely to induce hallucination than “is there a python package for <task>”. Essentially, the model is confused, if you’re asking for a python module, humans mean they want an existing module for x, but phrased as a command could be misunderstood to mean: invent a module for x.

## Model
We use LLaMA-2 7B: it’s open source, capable, but small enough to work with.

## Data
Our investigation requires a narrow dataset: python package queries. We use GPT-4o to generate prompt coding tasks. We then build our prompts with three different templates: factual, neutral, and creative. Prompts are fed into LLaMA-2 7B which generates responses. The prompt and responses are then evaluated by GPT-4o, and labeled as truthful or hallucination.

### Data Directory Organization
Data are organized in the following file structure:
`<coding-prompt-type>/<dataset-name>/<model>`



