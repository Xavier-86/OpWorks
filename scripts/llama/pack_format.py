"""OpWorks tensor pack container.

Single file layout:
    [8B magic "OPWPACK1"][8B uint64 LE json_len][json manifest][raw tensor blob]

Manifest:
    {
      "format_version": 1,
      "dtype": "float32",
      "tensors": {name: {"offset": int, "shape": [int, ...]}, ...},
      "aliases": {alias_name: target_name, ...},   # optional
      ...extra top-level keys (config, meta) allowed...
    }

Offsets are relative to the start of the raw blob and aligned to 64 bytes.
All tensors are little-endian float32, C-contiguous.
"""

import json
import struct

MAGIC = b"OPWPACK1"
ALIGN = 64


def _align(n, a=ALIGN):
    return (n + a - 1) // a * a


class PackWriter:
    def __init__(self):
        self.tensors = {}  # name -> (bytes, shape)
        self.aliases = {}

    def add_tensor(self, name, array):
        """array: numpy float32 array (any shape)."""
        import numpy as np

        arr = np.ascontiguousarray(array, dtype=np.float32)
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name}")
        self.tensors[name] = (arr.tobytes(), list(arr.shape))

    def add_tensor_bf16(self, name, tensor):
        """tensor: torch tensor, stored as raw little-endian bfloat16 bytes."""
        import numpy as np
        import torch

        raw = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name}")
        self.tensors[name] = (raw.tobytes(), list(raw.shape))

    def add_alias(self, alias, target):
        if target not in self.tensors:
            raise ValueError(f"alias target {target} missing")
        self.aliases[alias] = target

    def write(self, path, dtype="float32", extra=None):
        if dtype not in ("float32", "bfloat16"):
            raise ValueError(f"unsupported pack dtype {dtype}")
        manifest = {
            "format_version": 1,
            "dtype": dtype,
            "tensors": {},
        }
        if self.aliases:
            manifest["aliases"] = dict(self.aliases)
        if extra:
            manifest.update(extra)

        offset = 0
        for name, (data, shape) in self.tensors.items():
            manifest["tensors"][name] = {"offset": offset, "shape": shape}
            offset = _align(offset + len(data))

        blob = bytearray()
        for name, (data, shape) in self.tensors.items():
            blob += data
            blob += b"\x00" * (_align(len(blob)) - len(blob))

        js = json.dumps(manifest, sort_keys=True).encode("utf-8")
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(js)))
            f.write(js)
            f.write(bytes(blob))


def read_pack(path):
    """Returns (manifest_dict, {name: numpy float32 array})."""
    import numpy as np

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        (json_len,) = struct.unpack("<Q", f.read(8))
        manifest = json.loads(f.read(json_len))
        blob = f.read()

    out = {}
    for name, meta in manifest["tensors"].items():
        shape = meta["shape"]
        n = 1
        for d in shape:
            n *= d
        nbytes = n * 4
        off = meta["offset"]
        if off + nbytes > len(blob):
            raise ValueError(f"{name}: offset {off}+{nbytes} beyond blob {len(blob)}")
        out[name] = np.frombuffer(blob[off : off + nbytes], dtype="<f4").reshape(shape)
    for alias, target in manifest.get("aliases", {}).items():
        out[alias] = out[target]
    return manifest, out
