# FrequencyDiffusion
The repo for frequency diffusion.

Authors: Chenxi Wang

## Backgrounds
### DDPM
DDPM-related models rely on the forward process:
$$ \mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$
Here, $\alpha_t, \beta_t \in \mathbb{R}$ are scalers, indicating the all-dimension linear compression. 

### DDIM
DDIM 

### General Diffusions
Recently, many works try to extend the diffusion process into a more general way. Specifically, a general forward process is defined as follows:
$$ \mathbf{x}_t = D(\mathbf{x}_0, t) + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$
where $D(\cdot, t)$ is the degradation function, such as blurring, noising, and so on. In Cold Diffusion(NIPS2023), $\beta_t = 0$, while in Soft Diffusion(TMLR), $D(\mathbf{x}_0, t)= \mathbf{C}_t \mathbf{x}_0$.


## MovingAvg Diffusion
In time series forecasting, progressive (or coarse-to-fine) models can generally improve the accuracy. Such philosophy is highly aligned with Diffusion models. Meanwhile, most of However, noise-based diffusion model makes the intermediate products of diffusion model unusable, which are abandoned after sampling.  

Therefore, we try to explore the research problem: **Can we progressively produce usable forecasts through diffusion?**
<!-- In the field of time series forecasting, forecasts at different temporal resolution should follow the hierachical rule. It says that the temporally aggreagated high-resolution forecasts should be align with the low-resolution forecasts.  -->

<!-- In essense, such rule can be satified by moving average with different kernel on the highest-resolution forecasts. However, such naive trick only utilizes the information of the highest resolution data. We try to think  -->

### The choice of degradation function
In the most case of multi-resolution forecasting, low-resolution data/forecasts can be viewed as the temporally averaged results from high resolution ones. It implies that the degradation is moving average.

In terms of time domain, moving average can be expressed as convolution with the kernel $\mathbf{K}_t$:
$$ \mathbf{x}_t = \mathbf{K}_t * \mathbf{x}_0$$


In terms of frequency domain, moving average can be expressed as multiplication with the frequency response $\mathbf{\tilde{K}}_t$:
$$ \mathbf{\tilde{x}}_t = \mathbf{K}_t \cdot \mathbf{\tilde{x}}_0$$



