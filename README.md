# FrequencyDiffusion
The repo for frequency diffusion.

Authors: Chenxi Wang

## Backgrounds
### DDPM
DDPM-related models rely on the forward process:
$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$
Here, $\alpha_t, \beta_t \in \mathbb{R}$ are scalers, indicating the all-dimension linear compression. 

### DDIM
DDIM 

### General Diffusions
Recently, many works try to extend the diffusion process into a more general way. Specifically, a general forward process is defined as follows:
$$\mathbf{x}_t = D(\mathbf{x}_0, t) + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$
where $D(\cdot, t)$ is the degradation function, such as blurring, noising, and so on. In Cold Diffusion(NIPS2023), $\beta_t = 0$, while in Soft Diffusion(TMLR), $D(\mathbf{x}_0, t)= \mathbf{C}_t \mathbf{x}_0$.


## MovingAvg Diffusion
In time series forecasting, progressive (or coarse-to-fine) models can generally improve the accuracy. Such philosophy is highly aligned with Diffusion models. Meanwhile, noise-based diffusion model makes the intermediate products of diffusion model unusable, which are abandoned after sampling.  

Therefore, we try to explore the research problem: **Can we progressively produce usable forecasts through diffusion?**
<!-- In the field of time series forecasting, forecasts at different temporal resolution should follow the hierachical rule. It says that the temporally aggreagated high-resolution forecasts should be align with the low-resolution forecasts.  -->

<!-- In essense, such rule can be satified by moving average with different kernel on the highest-resolution forecasts. However, such naive trick only utilizes the information of the highest resolution data. We try to think  -->

### The choice of degradation function
In the most case of multi-resolution forecasting, low-resolution data/forecasts can be viewed as the temporally averaged results from high resolution ones. It implies that the degradation is moving average.

In terms of time domain, moving average can be expressed as convolution with the kernel $\mathbf{K}_t$:
$$\mathbf{x}_t = \mathbf{K}_t * \mathbf{x}_0 + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$


In terms of frequency domain, moving average can be expressed as multiplication with the frequency response $\mathbf{\tilde{K}}_t$:
$$\mathbf{\tilde{x}}_t = \mathbf{\tilde{K}}_t  \mathbf{\tilde{x}}_0 + \beta_t \tilde{\epsilon}, \tilde{\epsilon}  \sim \mathcal{CN}(0, \mathbf{I})$$

Here, we assume DFTs are all normalized by $\sqrt{1/N}$, $N$ is the sequence length. To unify two domain, we denote the moving average transform as $\mathbf{C}_t$.
### Sampling
The key of reverse process is the design of sampler:

$$\mathbf{\tilde{x}}_{t-1} = \sqrt{\frac{\beta_{t-1}^2 - \sigma_t^2}{\beta_t^2}} \mathbf{\tilde{x}}_t + \left(\mathbf{\tilde{K}}_{t-1} - \sqrt{\beta_{t-1}^2 - \frac{\sigma_t^2}{\beta_t^2}} \mathbf{\tilde{K}}_t\right) \mathbf{\tilde{x}}_0  + \sigma_t \epsilon^\prime, \epsilon^\prime \sim \mathcal{CN}(0, \mathbf{I})$$

<!-- $$\begin{aligned}
    q(\mathbf{\tilde{x}}_{t-1} | \mathbf{\tilde{x}}_t, \mathbf{\tilde{x}}_0) &= a \mathbf{\tilde{x}}_t + b \mathbf{\tilde{x}}_0  + \sigma_t \epsilon^\prime, \epsilon^\prime \sim \mathcal{C}\mathcal{N}(0, \mathbf{I}) \\
    &= a (\mathbf{\tilde{K}}_t \mathbf{\tilde{x}}_0 + \beta_t {\tilde{\epsilon}}) + b \mathbf{\tilde{x}}_0  + \sigma_t \epsilon^\prime \\
    &= (a \mathbf{\tilde{K}}_t + b) \mathbf{\tilde{x}}_0 + a \beta_t {\tilde{\epsilon}}  + \sigma_t \epsilon^\prime \\
    &= (a \mathbf{\tilde{K}}_t + b) \mathbf{\tilde{x}}_0 + \sqrt{a^2 \beta_t^2 + \sigma_t^2} \epsilon^*, \epsilon^* \sim \mathcal{CN}(0, \mathbf{I}) \\
    & = \mathbf{\tilde{K}}_{t-1}  \mathbf{\tilde{x}}_0 + \beta_{t-1} \tilde{\epsilon}
\end{aligned}$$

Therefore, we can design:
$$\begin{cases}
    a \mathbf{\tilde{K}}_t + b = \mathbf{\tilde{K}}_{t-1} \\
\sqrt{a^2 \beta_t^2 + \sigma_t^2} = \beta_{t-1}
\end{cases} \Rightarrow \begin{cases}
    a = \sqrt{{\beta_{t-1}^2 - \sigma_t^2}/{\beta_t^2}} \\
    b = \mathbf{\tilde{K}}_{t-1} - \sqrt{{\beta_{t-1}^2 - \sigma_t^2}/{\beta_t^2}} \mathbf{\tilde{K}}_t
\end{cases}$$ -->


## Experiments

### DDPM

| Method       | RMSE     | MAE      | CRPS     |
| ------------ | -------- | -------- | -------- |
| cnn_freq_2M  | 0.050387 | 0.041882 | 0.038450 |
| cnn_time_2M  | 0.040539 | 0.033057 | 0.031043 |
| mlp_freq_23M | 0.056778 | 0.047382 | 0.042925 |
| mlp_time_23M | 0.078413 | 0.065104 | 0.059050 |



<!-- | mlp_freq_700k (FAIL)       | 0.108959 | 0.089177 | 0.088174 |
| mlp_time_700k (FAIL)       | 0.103443 | 0.085088 | 0.073611 |
| freqlinear_time_60k (FAIL) | 0.107398 | 0.088021 | 0.084010 | -->

### MovingAvg

Deterministic sampling (DDIM, $\sigma_t = 0 $)

| Method       | RMSE     | MAE      | CRPS     |
| ------------ | -------- | -------- | -------- |
| cnn_freq_2M  | 0.042480 | 0.032064 | 0.037333 |
| cnn_time_2M  | 0.077175 | 0.055606 | 0.058438 |
| mlp_freq_5M  | 0.087949 | 0.065613 | 0.068308 |
| mlp_time_5M  | 0.070852 | 0.051746 | 0.054352 |

Stochastic sampling ($\sigma_t > 0 $)
need extra design


<!-- | cmlp_freq_5M | 0.074481 | 0.054804 | 0.057391 | -->
<!-- 
- [x] DDPM time
- [x] DDPM freq
- [x] MovingAvg time ($\beta_t =0$)
- [x] MovingAvg freq ($\beta_t =0$)
- [x] MovingAvg time
- [x] MovingAvg freq
 -->

 
