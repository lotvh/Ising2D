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

![energy_vs_field_time_ev1](https://user-images.githubusercontent.com/49079733/185738179-1e31a0d6-37fe-4335-832a-f2f092e8275f.png)

![energy_vs_field_time_ev2](https://user-images.githubusercontent.com/49079733/185738189-3b371a12-920c-4511-9a5a-ebb928ee1fbb.png)

![energy_vs_field_optimized1](https://user-images.githubusercontent.com/49079733/185738195-048bfb36-df3d-4b5c-bbe0-213780eb53d2.png)

![energy_vs_field_optimized2](https://user-images.githubusercontent.com/49079733/185738234-730444b5-f2de-45b5-b883-37caf3fa0879.png)
