# Algorithms to Compute the Mandelbrot Set
The algorithms in here are different ways to compute whether a complex number is part of the mandelbrot set or not.

A complex number $$c$$ is in the mandelbrot set if the following recurrence does not diverge to infinity when $$z_0 = 0$$:

$$z_(n+1) = z_n^2 + c$$

Since this is an infinite recurrence, no algorithm would be able to definitively tell if any complex number is in the mandelbrot set or not in 
a finite amount of time. Instead, these algorithms terminate at some amount of iterations and provide an approximate answer.

### Escape time
Checks whether a complex number "escapes" the range of the mandelbrot within a number of iterations. There are two implementations of this in the repo,
a sequential, CPU based one, and a CUDA based parallel one.

### Other algorithms
I am not currently planning on implementing these algorithms now, but they could be implemented in the future.

#### Buddhabrot
Tracks the trajectories of complex numbers as they iterate through the mandelbrot recurrence. This can reveal some hidden structure in the
mandelbrot recurrence not seen with the escape time algorithm.

#### Derivative bailout (Derbail)
Accumulates the derivatives of the mandelbrot recurrence through iterations until a bailout value. This can reveal structure in Julia sets, which
are closely related to the mandelbrot set.