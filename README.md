generalized-lloyd-quantization
==============================
This is a pure Python/NumPy/SciPy implementation of the Generalized Lloyd quantization scheme, 
both in the most basic formulation<sup>[[1]](#ref1)</sup><sup>[[2]](#ref1)</sup> and also in 
a formulation which is optimal in terms of entropy rate<sup>[[3]](#ref1)</sup><sup>[[4]](#ref1)</sup>.
The suboptimal version is often called the Linde Buzo Gray (LBG) algorithm and the optimal 
version is often called Entropy-Constrained Vector Quantization.

At some point I may add an implementation in a GPU computing framework so that it can scale more
easily for vector quantization in high dimensions, but for now this is an implementation that
should still be usable for a reasonable number of dimensions.

## Dependencies
* numpy
* scipy
* matplotlib
* [hdmedians](https://github.com/daleroberts/hdmedians)

## Example
Usage can be inferred from the example found in the demo/ folder. In this particular case we 
generate samples of a 2D multivariate random variable. We can either quantize each coefficient
separately or we can quantize the coefficients jointly. For this particular dataset it appears 
that not only are significant gains achieved by using the entropy-rate-optimal version of the 
algorithm, but that by quantizing the coefficients jointly, we also get a gain in coding 
efficiency.

![alt text](generalized_lloyd_quantization/demo/plots/composite_demo_fig.png)
![alt text](generalized_lloyd_quantization/demo/plots/rd_performance_example.png)

### Authors
Spencer Kent

### References
[<a name="ref1">1</a>]: Linde, Y., Buzo, A., & Gray, R. (1980). 
An algorithm for vector quantizer design. 
_IEEE transactions on information theory_, 28(1), 84-95.

[<a name="ref2">2</a>]: Lloyd, S. (1982).  Least squares quantization in PCM.  
_IEEE transactions on information theory_, 28(2), 129-137.

[<a name="ref3">3</a>]: Berger, T. (1982). 
Minimum entropy quantizers and permutation codes. 
_IEEE transactions on information theory_, 28(2), 149-157.

[<a name="ref3">4</a>]: Chou, P. A., Lookabaugh, T., & Gray, R. M. (1989). 
Entropy-constrained vector quantization. 
_IEEE transactions on acoustics, speech, and signal processing_, 37(1), 31-42.
