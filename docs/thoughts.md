# Chat with Nicolas C. (26/06/2018)

* <a href="../media/20180626_151928.jpg">20180626_151928.jpg</a>
* <a href="../media/20180626_151939.jpg">20180626_151939.jpg</a>

Let us consider the early classification setting in which one has to minimize the following per-instance loss:

$$L(x, y, \{h_t\}, \tau) = L_c(h_\tau(x), y) + \alpha \tau(x)$$

where $\tau$ is the time at which the classifier performs classification, which has to be learned together with 
parameters for $\{h_t\}$.

We would like to tackle this setting using a single classifier $h$ for all time steps, contrary to what is done in 
Dachraoui's paper (ECML 2015) or in ours (ECML 2016).
To do so, Recurrent Neural Nets appear as a quite reasonable solution.

This chat has lead to two alternatives for this problem.

## Option 1 : Training two Recurrent Neural Nets sequentially

In this approach, the idea would be to:

1. train a first RNN $h$ that would aim at classifying pieces of time series in a 
many-to-one fashion. 
2. Based on this trained classifier $h$, one could build, for each training time series $x$ with 
label $y$, a loss time series that would associate, to each time instant $t$, the loss $L(x, y, h, t)$.
3. Finally, deciding on the time $\tau$ at which classification should be performed could be casted as a regression 
problem in which, at each time instant $t$, the quantity to be regressed on is the remaining waiting time before 
classification, denoted as $\Delta_t$. In such a setting, ground truth data could be derived from $L(t)$ time series 
built at step 2: the target $\Delta_t$ being computed as the time delta between $t$ and the argmin of $L(t)$ in the 
future. If $\Delta_t = 0$, classification should be performed at time $t$ whereas if $\Delta_t \gt 0$, it should be
postponed (and $\Delta_t$ would provide an educated guess of the time one would have to wait before classification).
 
Two (sub-)options could be considered for step 1. 
First, a very standard RNN could be used in which, at time $t$, one would pass $x_0, 
\dots x_t$ through the system and output classification probabilities at time $t$.
A possible limitation of that approach is that the model would not be able to learn "not to decide" for early timings.
A way to alleviate this burden would be to have a **mixed generative-discriminant model**.
The idea here would be to generate, at time $t$, a series of potential ends $\hat{x}_{t+1}, 
\dots \hat{x}_T$ for our time series $x_0, \dots x_t$ and average classification probabilities over all completed 
sequences $x_0, \dots \hat{x}_T$. This way, early in the process, if we are likely to generate very different ends, we
will probably get very blurred probabilities, contrary to what we would get for late $t$ values.
One possible issue here is that the number of ends to draw should be chosen with great care and would probably be 
related to the coefficient $\alpha$ for the temporal cost.

The cost function to be minimized for this mixed generative-discriminant model would look like:

$$E_x E_t E_{\hat{x}_{t+1}, \dots \hat{x}_T \text{ draws}} L_c(h(x_0, \dots \hat{x}_T), y) + 
\lambda \| (x_0, \dots \hat{x}_T) - (x_0, \dots x_T) \|^2_2$$


## Option 2 : All-in-one-RNN

Another, more challenging, option would be to learn a model that would have two discriminative outputs plugged on the
recurrent unit at time $t$ :
1. An output $h_t$ that would output class probabilities as usual
2. An output $d_t$ that would output a probability to delay classification: if this value is lower than a half, more 
data has to be obtained before classification, otherwise classification is performed.

The cost function to be minimized here would be:

$$E_x E_t P(t) * \left(L_c(h_t(x_0, \dots x_t), y) + \alpha t\right)$$

where $P(t)$ would be computed as:

$$P(0) = d_0(x_0)$$

**TODO: For other t values, could we derive a formulation?**

$$P(T) = 1 - \sum_{t<T} P(t)$$

# Chat with Rémi E. (20/07/2018)

* <a href="../media/20180720_114608.jpg">20180720_114608.jpg</a>

In fact for **option 2** above, we have:

$$P(t) = d_t(x_t) \Pi_{t'<t}(1-d_{t'}(x_{t'}))$$

In practice, enforcing the constraint on $P(T)$ could be done through regularization using a term such as:

$$\left(1 - \sum_{t=0}^T P(t)\right)^2$$

Then we still have one question: at test time, how should we decide to classify or not?
One option would be to flip a coin based on $d_t(x_t)$ and classify if the coin says so.

We have also discussed a variant related to **option 1** above:

The idea would be to have a single RNN trained alternatively using 2 subsets of the training set:

1. Set #1 is used to train the classifier part of the mixed generative-discriminant model
2. Given a fixed classifier, the set #2 is used to train the generator that would generate cost curves (built from 
the freezed discriminator part)

This means that we would need to decide on when to classify.
For that, one option would be to draw several cost curves and decide based on quantiles rather than just on one draw or 
an average of all draws.