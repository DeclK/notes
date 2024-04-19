# VideoMAE

[arxiv](https://arxiv.org/abs/2203.12602) [github](https://github.com/MCG-NJU/VideoMAE)

## Concept

- An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance for VideoMAE

- VideoMAE achieves impressive results on very small datasets

- VideoMAE shows that data quality is more important than data quantity for SSVP

  We demonstrate that VideoMAE is a data-efficient learner that could be successfully trained with only 3.5k videos

  ![image-20240303172210920](VideoMAE/image-20240303172210920.png)

- Self-supervised v.s. supervised

  The learned representations have outperformed the ones via supervised learning when being transferred to downstream tasks.

  It is expected that this self-supervised learning paradigm can provide a promising solution to address the challenge of training video transformers.

- vannila mae is not sufficient to cover the image, because it can find the original information at different frames easily

  **Temporal redundancy & correlation. **

- space-time cube embedding

  T x H x W

  This design can decrease the spatial and temporal dimension of input, which helps to alleviate the spatiotemporal redundancy in videos.

  **High Sematic!**

- when we try to transfer the pre-trained VideoMAE models to the other video datasets (e.g. from Kinetics to Something-Something), the results are slightly worse than their counterpart, which is directly pre-trained on its own target video datasets.

## Question

- VideoGPT v.s. VideoMAE

- TSN Temporal segment networks for action recognition in videos

  如何进行动作识别