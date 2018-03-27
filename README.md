# Basic Implementation of a Continuous-attractor Neural Network

This repository stores a simple implementation of a continuous-attractor neural network model used in Fung, Wong and Wu (2008) and Fung, Wong and Wu (2010). The equation used in the python code was not exactly the same as that in those papers. This is a rescaled version:

\tau \frac{du(x,t)}{dt} = -u(x,t) + \int dx^\prime J\left(x,x^\prime\right) r\left(x^\prime,t\right)+A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right]

r(x,t) = \frac{\left[u(x,t)\right]_+{}^2}{1+\frac{k}{8\sqrt{2\pi}a}\int dx^\prime \left[u(x^\prime,t)\right]_+{}^2}

J\left(x,x^\prime\right) = \frac{1}{\sqrt{2\pi}a}\exp\left(-\frac{\left|x-x^\prime\right|^2}{2a^2}\right)

Except the dynamic variable and parameters are rescaled versions, the magnitude of external input is now decoupled from the magnitude of $u$ in stationary states.

In the simulation, how the change of network states responding to change of external input is demonstrated. In which, the external input is initially at $z=0$ in the preferred stimuli space. At $t=0$, the external input was changed from 0 to $z_0$. Snapshots taken per $10 \tau$ are presented to show the transient of the process.

Parameters:                                                    

-k [float] : Degree of the rescaled inhibition                           
-a [float] : Half-width of the range of excitatory connections           
-N [int] : Number of neurons / units                                   
-A [float] : Magnitude of the rescaled external input                   
-z0 [float] : New position of the external input after the sudden change 

