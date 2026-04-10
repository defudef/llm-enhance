# Gemma 4 E2B Controller MVP

## Goal

Build a separate controller for `Gemma 4 E2B` without modifying the base model
weights.

This cannot reuse the current Trinity controller directly:

- the current controller is a `router-bias` controller for MoE routing
- `Gemma 4 E2B` is a dense model, not a router-driven MoE model
- the controller therefore needs to steer the model through the input path or
  hidden states, not through expert selection

## Proposed MVP

Use a `soft-prompt controller`.

High-level flow:

1. freeze the base Gemma model
2. embed the user prompt with the base input embeddings
3. pool the prompt embeddings into a fixed-size representation
4. map that representation to `N` virtual token embeddings
5. prepend those virtual token embeddings to the original prompt embeddings
6. run normal causal LM training or inference with the augmented input

This is the lowest-risk dense-controller design because:

- it does not depend on MoE internals
- it is compatible with `inputs_embeds` in Hugging Face models
- it keeps the base model frozen
- it is small enough to train cheaply

## Why This Fits Gemma 4 E2B

`Gemma 4 E2B` is presented by Google as a dense model in the Gemma 4 family,
with the smaller `E` models using parameter-efficient embedding mechanisms
instead of MoE routing. That means the right steering point is the prompt/input
path, not the expert router.

## Controller Shape

Input:

- `prompt_embeds`: `[batch, prompt_tokens, hidden_size]`
- `attention_mask`: optional `[batch, prompt_tokens]`

Output:

- `soft_prompt_embeds`: `[batch, num_virtual_tokens, hidden_size]`

Recommended MVP config:

- `num_virtual_tokens = 8` or `16`
- `controller_dim = 256`
- `hidden_dim = 1024`
- `dropout = 0.0`

## Controller Architecture

MVP module:

1. masked mean-pool prompt embeddings
2. apply layer norm
3. MLP: `hidden_size -> hidden_dim -> controller_dim`
4. projection head:
   `controller_dim -> num_virtual_tokens * hidden_size`
5. reshape to `[batch, num_virtual_tokens, hidden_size]`
6. add a learned base prompt table for stability

The learned base prompt table lets the controller start from a stable prompt
bias and learn per-input deltas.

## Inference Integration

For Hugging Face Gemma text-only usage:

1. tokenize the prompt normally
2. get base token embeddings from the frozen model
3. compute controller soft prompts
4. prepend controller outputs to `inputs_embeds`
5. extend `attention_mask` with ones for the virtual tokens
6. call the model with `inputs_embeds=...`

Important:

- labels during training should ignore the prepended virtual token positions
- the generation path should use the soft prompt only on the initial forward
  pass, then continue with KV cache normally

## Training Objective

Start with standard next-token loss on the target response:

- freeze base Gemma
- train only the controller
- ignore prompt tokens in labels, same idea as current Trinity controller

Optional regularization:

- L2 on controller output magnitude
- cosine penalty against exploding soft prompt norms

## Evaluation Plan

Recommended order:

1. tiny smoke dataset to verify the training loop
2. small style/control dataset to see whether the controller can steer format
3. `IFEval` as the first real benchmark

Primary metrics:

- `IFEval` strict instruction accuracy
- `IFEval` loose instruction accuracy
- latency overhead vs frozen base model

## Future Variants

If the soft-prompt controller is too weak, the next upgrade should be:

- hidden-state steering at selected decoder layers

Do not start there. It is more invasive and harder to debug than soft prompts.

## Implementation Notes For This Repo

The first implementation step in this repository is:

- add a model-agnostic `PromptPoolingSoftPromptController`
- add a helper that prepends controller embeddings to `inputs_embeds`
- keep Gemma integration separate from Trinity runtime

This keeps the controller reusable even if we later test:

- Gemma 4 E2B
- Gemma 4 E4B
- other dense decoder models from Transformers
