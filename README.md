# STDWI

Implementation of a spike timing-dependent weight Inference (STDWI) method and competitors -- all of which are proposed as biologically plausible methods to solve the weight transport problem for the backpropagation of error algorithm.

In this repository we have our implementation of the STDWI method, the regression discontinuity design (RDD) method by Guerguiev et al. and a modified rate-based method by Akrout et al.
See [Example.ipynb](./Example.ipynb)) for a walkthrough of simulating a feedforward network of leaky integrate and fire neurons and inference of the synaptic weights using these techniques.

The scripts used to produce plots shown in our [arXiv pre-print]() are located in the [paper_scripts](./paper_scripts/) folder.

Guerguiev, J., Kording, K. P., & Richards, B. A. (2019). Spike-based causal inference for weight alignment. In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/1910.01689

Akrout, M., Wilson, C., Humphreys, P. C., Lillicrap, T., & Tweed, D. (2019). Deep Learning without Weight Transport. In arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1904.05391
