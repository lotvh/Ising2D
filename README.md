# How to run

-   Clone this repository
-   In the location of the cloned repository, and with the docker service running, run:

```shell
$ docker build -t [name of the image] .
```

-   Then run

```shell
$ docker run [name of the image]
```

# Methods

There are two methods available:
- `simple_time_evolution`: simplified algorithm using two dimensional time evolving block decimation
- `autodiff_opt`: a global autodiff optimization method using gradients and a cost function.

The first method is more stable but less accurate than the second one, which can also be seen in
the energy vs field plots:
