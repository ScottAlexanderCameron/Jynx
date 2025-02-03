from dataclasses import dataclass, field, fields
from typing import dataclass_transform

import jax.tree_util as tu


def static(*args, **kwargs):
    """Marks fields in a dataclass as static for JAX pytree processing.

    This function is a utility to mark certain fields of a dataclass as
    static, meaning they will not be considered part of the tree structure
    by JAX operations. Fields marked as static are stored in the `aux`
    data structure and remain unchanged by JAX transformations.

    Args:
        *args: Positional arguments passed to the `field` constructor.
        **kwargs: Keyword arguments passed to the `field` constructor.

    Returns:
        A dataclass field with the `static` metadata set, indicating it
        should be treated as a static attribute.

    """
    return field(*args, **kwargs, metadata={"static": True})


def dataclass_flatten(self):
    """Flattens a dataclass instance for JAX pytree processing.

    Separates the dataclass attributes into two categories: dynamic
    and static. Dynamic attributes are considered part of the tree
    structure and are returned for further processing by JAX, while
    static attributes are stored separately and remain unchanged.

    Returns:
        A tuple containing a single-element tuple with the dynamic
        attributes, and a dictionary of static attributes.

    """
    fs = {f.name: f.metadata.get("static", False) for f in fields(self)}
    ch = {name: getattr(self, name) for name, st in sorted(fs.items()) if not st}
    static = {name: getattr(self, name) for name, st in sorted(fs.items()) if st}
    return ch.values(), (ch.keys(), static)


def dataclass_unflatten(cls, aux, children):
    """Reconstructs a dataclass instance from flattened dynamic and static attributes.

    This is the inverse operation of `dataclass_flatten`, used by JAX
    to reconstruct the dataclass instance after processing.

    Args:
        cls: The dataclass type to be reconstructed.
        aux: A dictionary of static attributes.
        children: A tuple containing the dynamic attributes.

    Returns:
        An instance of the dataclass with attributes reconstructed from
        the provided dynamic and static data.

    """
    keys, static = aux
    children = dict(zip(keys, children, strict=True))
    return cls(**children, **static)


def dataclass_flatten_with_keys(self):
    """Flattens a dataclass instance with keys for JAX pytree processing.

    An extension of `dataclass_flatten` that includes keys for each
    dynamic attribute, providing more control and clarity over the
    flattening process. This is particularly useful when the order or
    identity of attributes is significant.

    Returns:
        A list of tuples, each containing an attribute key and value,
        and a dictionary of static attributes.

    """
    ch, (keys, static) = dataclass_flatten(self)
    return [(tu.GetAttrKey(k), v) for k, v in zip(keys, ch, strict=True)], (
        keys,
        static,
    )


@dataclass_transform(field_specifiers=(static,))
class PyTree:
    """A base class for network layers, making them compatible with JAX pytree operations.

    `PyTree` is designed to be subclassed by network layer classes,
    automatically converting them into dataclasses and registering them
    as pytree types with JAX. This registration enables instances of
    these classes to be seamlessly used with JAX functions like `jit`,
    `grad`, and `vmap`, allowing JAX to process their attributes as if
    they were native JAX types.

    The class provides methods for flattening and unflattening instances,
    compatible with JAX's expectations for pytree nodes. This allows
    custom data structures to be integrated into JAX workflows, supporting
    automatic differentiation and JIT compilation of functions that
    operate on complex data structures.

    Example:
        ```python
        class MyLayer(PyTree):
            weights: jax.Array
            biases: jax.Array

        # Now `MyLayer` instances can be used with JAX functions
        layer = MyLayer(weights=jnp.ones((10, 10)), biases=jnp.zeros(10))
        grad_fn = jax.grad(some_loss_function)
        gradients = grad_fn(layer)
        ```

    Note:
        Fields can be marked as static using the `static` function,
        indicating that they should not be considered as part of the
        tree structure for the purposes of JAX transformations.

    Attributes:
        Any attribute defined in subclasses will be automatically handled
        as part of the dataclass and, unless marked as static, will be
        considered dynamic attributes for JAX processing.

    """

    def __init_subclass__(cls: type):
        super().__init_subclass__()
        return tu.register_pytree_with_keys_class(dataclass(frozen=True)(cls))

    tree_flatten_with_keys = dataclass_flatten_with_keys
    tree_flatten = dataclass_flatten
    tree_unflatten = classmethod(dataclass_unflatten)
