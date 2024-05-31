# MAD: Moving Average Diffusion
The repo for Moving Average Diffusion

<!-- Authors: Chenxi Wang -->

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


## Moving Average Diffusion
In time series forecasting, progressive (or coarse-to-fine) models can generally improve the accuracy. Such philosophy is highly aligned with the one of Diffusion models. However, noise-based diffusion model makes the intermediate products of diffusion model unusable, which are abandoned after sampling.  

Therefore, we try to explore the research problem: **Can we progressively produce usable forecasts through diffusion?**
<!-- In the field of time series forecasting, forecasts at different temporal resolution should follow the hierachical rule. It says that the temporally aggreagated high-resolution forecasts should be align with the low-resolution forecasts.  -->

<!-- In essense, such rule can be satified by moving average with different kernel on the highest-resolution forecasts. However, such naive trick only utilizes the information of the highest resolution data. We try to think  -->

### The choice of degradation function
In the most case of multi-resolution forecasting, low-resolution data/forecasts can be viewed as the temporally averaged results from high resolution ones. It implies that the degradation is moving average.

In terms of time domain, moving average can be expressed as convolution with the kernel $\mathbf{K}_t$:
$$\mathbf{x}_t = \mathbf{K}_t * \mathbf{x}_0 + \beta_t \epsilon, \epsilon  \sim \mathcal{N}(0, \mathbf{I})$$
$\mathbf{x}_0$ is a windowed time series data in the original resolution.


In terms of frequency domain, moving average can be expressed as multiplication with the frequency response $\mathbf{\tilde{K}}_t$:
$$\mathbf{\tilde{x}}_t = \mathbf{\tilde{K}}_t  \mathbf{\tilde{x}}_0 + \beta_t \tilde{\epsilon}, \tilde{\epsilon}  \sim \mathcal{CN}(0, \mathbf{I})$$

Here, we assume DFTs are all normalized by $\sqrt{1/N}$, $N$ is the sequence length.
### Sampling
The key of reverse process is the design of sampler, and here we design a DDIM-type sampling process.
$$\mathbf{\tilde{x}}_{t-1} = \sqrt{\frac{\beta_{t-1}^2 - \sigma_t^2}{\beta_t^2}} \mathbf{\tilde{x}}_t + \left(\mathbf{\tilde{K}}_{t-1} - \sqrt{\frac{\beta_{t-1}^2 - \sigma_t^2}{\beta_t^2}} \mathbf{\tilde{K}}_t\right) \mathbf{\tilde{x}}_0  + \sigma_t \epsilon^\prime, \epsilon^\prime \sim \mathcal{CN}(0, \mathbf{I})$$

When $\sigma_t = 0$, we have the deterministic sampling schedule:
$$\frac{\mathbf{\tilde{x}}_{t-1}}{\beta_{t-1}} = \frac{\mathbf{\tilde{x}}_t}{\beta_t}  + \left(\frac{\mathbf{\tilde{K}}_{t-1}}{\beta_{t-1}} - \frac{\mathbf{\tilde{K}}_t}{\beta_t^2}\right) \mathbf{\tilde{x}}_0$$

<!-- 
$$\begin{aligned}
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
    a = \sqrt{(\beta_{t-1}^2 - \sigma_t^2)/{\beta_t^2}} \\
    b = \mathbf{\tilde{K}}_{t-1} - \sqrt{(\beta_{t-1}^2 - \sigma_t^2)/{\beta_t^2}} \mathbf{\tilde{K}}_t
\end{cases}$$ -->


## Experiments

<!-- ### DDPM

| Method       | RMSE     | MAE      | CRPS     |
| ------------ | -------- | -------- | -------- |
| cnn_freq_2M  | 0.050387 | 0.041882 | 0.038450 |
| cnn_time_2M  | 0.040539 | 0.033057 | 0.031043 |
| mlp_freq_23M | 0.056778 | 0.047382 | 0.042925 |
| mlp_time_23M | 0.078413 | 0.065104 | 0.059050 | -->


### Benchmark
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

### MovingAvg Diffusion

<!-- Linear schedule, deterministic sampling, fast_sample -->
Test test
NHITS from AAAI     0.038xxxx
Transformers      > 0.04xxxx

training not so 
sampling


| method                     | RMSE     | MAE      | CRPS     |
| -------------------------- | -------- | -------- | -------- |
| mlp_freq_norm_on_diff_on   | 0.032953 | 0.024754 | 0.029487 |
| mlp_freq_norm_on_diff_off  | 0.035471 | 0.026272 | 0.029652 |
| mlp_time_norm_on_diff_on   | 0.035325 | 0.026056 | 0.030248 |
| mlp_time_norm_on_diff_off  | 0.036394 | 0.026610 | 0.030867 |
| cnn_freq_norm_on_diff_on   | 0.035590 | 0.026405 | 0.031303 |
| cnn_freq_norm_on_diff_off  | 0.032834 | 0.024736 | 0.030798 |
| cnn_time_norm_on_diff_on   | 0.036176 | 0.026803 | 0.031058 |
| cnn_time_norm_on_diff_off  | 0.035221 | 0.026047 | 0.030936 |


| method                     | RMSE     | MAE      | CRPS     |
| -------------------------- | -------- | -------- | -------- |
| mlp_freq_norm_off_diff_on  | 0.085846 | 0.065210 | 0.067466 |
| mlp_freq_norm_off_diff_off | 0.084568 | 0.062234 | 0.064756 |
| mlp_time_norm_off_diff_on  | 0.089474 | 0.065595 | 0.067332 |
| mlp_time_norm_off_diff_off | 0.071982 | 0.052443 | 0.055350 |
| cnn_freq_norm_off_diff_on  | 0.090144 | 0.065718 | 0.068588 |
| cnn_freq_norm_off_diff_off | 0.043629 | 0.032828 | 0.037774 |
| cnn_time_norm_off_diff_on  | 0.073314 | 0.053837 | 0.056869 |
| cnn_time_norm_off_diff_off | 0.081367 | 0.057613 | 0.060580 |


<!-- Constant schedule($\beta_t$=0, cold), deterministic sampling, fast_sample
| method                                     | RMSE     | MAE      | CRPS     |
| ------------------------------------------ | -------- | -------- | -------- |
| MLPBackbone_freq_norm_True_diff_False_cold | 0.056430 | 0.042523 | 0.049077 |
| MLPBackbone_freq_norm_True_diff_True_cold  | 0.055974 | 0.042676 | 0.049130 |
| MLPBackbone_time_norm_True_diff_False_cold | 0.045785 | 0.034590 | 0.040869 |
| MLPBackbone_time_norm_True_diff_True_cold  | 0.048523 | 0.037343 | 0.043683 |

Constant schedule($\beta_t$=1, hot), deterministic sampling, fast_sample
| MLPBackbone_freq_norm_True_diff_False_hot  | 0.035727 | 0.026662 | 0.029548 |
| MLPBackbone_freq_norm_True_diff_True_hot   | 0.033309 | 0.024634 | 0.027158 |
| MLPBackbone_time_norm_True_diff_False_hot  | 0.032919 | 0.023763 | 0.026949 |
| MLPBackbone_time_norm_True_diff_True_hot   | 0.034374 | 0.025083 | 0.028804 |

Stochastic sampling ($\sigma_t > 0$, Linear schedule, [0.01, 0.1])

<!-- 1. small noise level: max = 1e-1 -->

<!-- | Method   | RMSE     | MAE      | CRPS     |
| -------- | -------- | -------- | -------- |
| cnn_freq | 0.034253 | 0.025336 | 0.030797 |
| cnn_time | 0.060331 | 0.043948 | 0.046540 |
| mlp_freq | 0.069998 | 0.051043 | 0.053416 |
| mlp_time | 0.057428 | 0.041764 | 0.043979 |
| DLinear  | 0.089804 | 0.073082 | -        | --> -->



## TODO
- [ ] instance normalization -- bug fixing
- [ ] $\sigma_t$ design -- cosine schedules
- [ ] check the intermediate products (muli-resolution forecasts)
- [ ] other SOTA forecasting method -- DLinear
- [ ] embedding $t$ multiplication

<!-- - [ ] $\beta_t = \mathbf{\tilde{K}}_t$ -->




