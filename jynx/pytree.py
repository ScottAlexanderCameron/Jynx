from dataclasses import dataclass, field, fields

from typing_extensions import dataclass_transform

import jax.tree_util as tu


def static(*args, **kwargs):
    return field(*args, **kwargs, metadata={"static": True})


def dataclass_flatten(self):
    fs = {f.name: f.metadata.get("static", False) for f in fields(self)}
    ch = {name: getattr(self, name) for name, st in sorted(fs.items()) if not st}
    aux = {name: getattr(self, name) for name, st in sorted(fs.items()) if st}
    return (ch,), aux


def dataclass_unflatten(cls, aux, children):
    (children,) = children
    return cls(**children, **aux)


def dataclass_flatten_with_keys(self):
    (ch,), aux = self.tree_flatten()
    return [
        (tu.GetAttrKey(k), v) for k, v in sorted(ch.items(), key=lambda a: a[0])
    ], aux


@dataclass_transform(field_specifiers=(static,))
class PyTree:
    def __init_subclass__(cls: type):
        return tu.register_pytree_with_keys_class(dataclass(frozen=True)(cls))

    tree_flatten_with_keys = dataclass_flatten_with_keys
    tree_flatten = dataclass_flatten
    tree_unflatten = classmethod(dataclass_unflatten)
