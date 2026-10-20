"""The only additional learned module: 3 -> 1024 -> condition width."""
import flax.nnx as nnx
import jax


class SpatialMLP(nnx.Module):
    def __init__(self, output_dim: int, *, rngs: nnx.Rngs):
        self.input = nnx.Linear(3, 1024, rngs=rngs)
        self.output = nnx.Linear(
            1024, output_dim, rngs=rngs,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, displacement):
        return self.output(jax.nn.relu(self.input(displacement)))
