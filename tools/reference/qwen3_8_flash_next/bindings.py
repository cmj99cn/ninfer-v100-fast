from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from tools.artifact.layouts import align_up


class BindingError(ValueError):
    pass


@dataclass(frozen=True)
class Volume:
    path: Path
    payload_start: int
    payload_bytes: int
    logical_start: int


@dataclass(frozen=True)
class V3Object:
    id: str
    kind: str
    shape: tuple[int, ...]
    format: str
    layout: str
    offset: int
    bytes: int

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class V3Part:
    object: str
    begin: int
    end: int


class V3Artifact:
    def __init__(self, path):
        self.path = Path(path)
        self._fds = {}
        with open(self.path, "rb") as entry:
            header = entry.read(32)
            if header[:8] != b"NINFER\x00\x03":
                raise BindingError("not a NInfer v3 entry file")
            json_bytes = struct.unpack("<Q", header[8:16])[0]
            directory = json.loads(entry.read(json_bytes))
        self.directory = directory
        self.metadata = directory.get("metadata", {})
        if self.metadata.get("name") != "qwen3.8-flash-next":
            raise BindingError("unexpected artifact identity " + repr(self.metadata.get("name")))
        self.config = directory["components"]["text"]["config"]
        self.resources = directory["components"]["text"].get("resources", {})
        self.objects = {obj["id"]: self._object(obj) for obj in directory["objects"]}
        self.bindings = directory["bindings"]
        self.uses = directory.get("uses", [])
        entry_start = align_up(32 + json_bytes, 4096)
        files = directory["files"]
        volumes = []
        cursor = 0
        for index, member in enumerate(files):
            payload_bytes = int(member["payload_bytes"])
            if index == 0:
                path = self.path
                start = entry_start
            else:
                path = self.path.parent / member["path"]
                start = 4096
            volumes.append(Volume(path, start, payload_bytes, cursor))
            cursor += payload_bytes
        self.volumes = volumes
        self.payload_bytes = cursor

    @staticmethod
    def _object(obj):
        if obj.get("kind") == "tensor":
            return V3Object(
                obj["id"],
                "tensor",
                tuple(int(dim) for dim in obj["shape"]),
                obj["format"],
                obj["layout"],
                int(obj["offset"]),
                int(obj["bytes"]),
            )
        return V3Object(obj["id"], obj.get("kind", "resource"), (), obj.get("format", ""),
                        obj.get("layout", ""), int(obj["offset"]), int(obj["bytes"]))

    def _fd(self, path):
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read_logical(self, offset, count):
        if offset < 0 or count < 0 or offset + count > self.payload_bytes:
            raise BindingError("logical read outside the payload space")
        chunks = []
        remaining = count
        position = offset
        for volume in self.volumes:
            begin = volume.logical_start
            end = begin + volume.payload_bytes
            if position >= end or remaining == 0:
                continue
            if position < begin:
                raise BindingError("logical read has a gap")
            take = min(remaining, end - position)
            file_offset = volume.payload_start + (position - begin)
            data = os.pread(self._fd(volume.path), take, file_offset)
            if len(data) != take:
                raise BindingError("short artifact read")
            chunks.append(data)
            position += take
            remaining -= take
        if remaining != 0:
            raise BindingError("logical read exceeds the volumes")
        return b"".join(chunks)

    def object(self, object_id):
        try:
            return self.objects[object_id]
        except KeyError as error:
            raise BindingError("missing object " + object_id) from error

    def read_object(self, object_id, begin=0, count=None):
        obj = self.object(object_id)
        if count is None:
            count = obj.bytes - begin
        return self.read_logical(obj.offset + begin, count)

    def object_bytes(self, object_id):
        return self.read_object(object_id)

    def parts(self, name):
        binding = self.bindings.get(name)
        if binding is None:
            raise BindingError("missing binding " + name)
        if "object" in binding:
            obj = self.object(binding["object"])
            elements = 1
            for dim in obj.shape:
                elements *= dim
            return [V3Part(binding["object"], 0, elements)]
        result = []
        for part in binding["parts"]:
            result.append(V3Part(part["object"], int(part["range"][0]), int(part["range"][1])))
        return result

    def single_object(self, name):
        parts = self.parts(name)
        object_ids = {part.object for part in parts}
        if len(object_ids) != 1:
            raise BindingError(name + " spans several objects")
        return next(iter(object_ids))

    def resource(self, role):
        object_id = self.resources.get(role)
        if object_id is None:
            raise BindingError("missing resource role " + role)
        return self.object_bytes(object_id)

    def ple_hash_controls(self):
        raw = self.resource("ple_hash_controls")
        if len(raw) != 280:
            raise BindingError("the PLE hash controls are not 280 bytes")
        words = struct.unpack("<35Q", raw)
        return words[0:3], words[3:19], words[19:35]
