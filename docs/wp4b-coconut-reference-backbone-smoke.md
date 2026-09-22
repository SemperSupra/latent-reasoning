# WP4b Coconut reference-backbone CPU smoke

This preflight uses the exact pinned official Coconut core at
`facebookresearch/coconut@27273cb8cca4bb763c041a63b036d0c3b7cbbb48`
with the reference GSM8K backbone named by upstream:
`openai-community/gpt2`.

It performs one forward/backward/optimizer step on CPU with two latent slots.
The purpose is to retire compatibility risk between the upstream Coconut core
and the actual reference backbone before using sovereign GPUs.

This is not a paper reproduction: it does not use the GSM8K training data,
stage curriculum, checkpoint selection, or reference distributed launcher.
