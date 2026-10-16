"""Dependency-free reader for the subset of the ONNX protobuf schema that
operator-coverage discovery needs.

Only these fields are decoded::

    ModelProto
        8 : OperatorSetIdProto  opset_import   (repeated)
        7 : GraphProto          graph
    GraphProto
        1 : NodeProto           node           (repeated)
    NodeProto
        1 : string              input          (repeated)
        2 : string              output         (repeated)
        3 : string              name
        4 : string              op_type
        7 : string              domain
    OperatorSetIdProto
        1 : string              domain
        2 : int64               version        (varint)

Unknown fields are skipped, so models produced by newer ONNX releases keep
working.  No third-party protobuf runtime is required, which keeps the tool
runnable on a bare CI image.
"""

from dataclasses import dataclass, field

DEFAULT_DOMAIN = "ai.onnx"

_GROUP_WIRE_TYPES = (3, 4)


class MalformedProtobuf(ValueError):
    """Raised when a byte string cannot be decoded as a protobuf message."""


def normalize_domain(domain):
    """Map the empty protobuf default domain onto ONNX's canonical name."""
    return DEFAULT_DOMAIN if domain in ("", "ai.onnx") else domain


def _text(raw):
    return raw.decode("utf-8", "replace")


def _read_varint(data, pos):
    shift = 0
    value = 0
    size = len(data)
    while True:
        if pos >= size:
            raise MalformedProtobuf("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value >= 1 << 63:
                value -= 1 << 64
            return value, pos
        shift += 7
        if shift > 70:
            raise MalformedProtobuf("varint too long")


def _read_bytes(data, pos):
    length, pos = _read_varint(data, pos)
    end = pos + length
    if length < 0 or end > len(data):
        raise MalformedProtobuf("truncated length-delimited field")
    return data[pos:end], end


def iter_fields(data):
    """Yield ``(field_number, wire_type, value)`` tuples for one message.

    ``value`` is an ``int`` for varints and ``bytes`` for all other supported
    wire types.  Nested messages are returned as undecoded bytes and are
    decoded lazily by the caller.
    """
    pos = 0
    size = len(data)
    while pos < size:
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if field_number == 0:
            raise MalformedProtobuf("field number 0 is invalid")
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
        elif wire_type == 1:
            if pos + 8 > size:
                raise MalformedProtobuf("truncated 64-bit field")
            value = data[pos:pos + 8]
            pos += 8
        elif wire_type == 2:
            value, pos = _read_bytes(data, pos)
        elif wire_type == 5:
            if pos + 4 > size:
                raise MalformedProtobuf("truncated 32-bit field")
            value = data[pos:pos + 4]
            pos += 4
        elif wire_type in _GROUP_WIRE_TYPES:
            raise MalformedProtobuf("group wire types are not supported")
        else:
            raise MalformedProtobuf("unknown wire type %d" % wire_type)
        yield field_number, wire_type, value


@dataclass(frozen=True)
class Node:
    op_type: str
    domain: str
    name: str
    inputs: tuple = field(default=())
    outputs: tuple = field(default=())


@dataclass(frozen=True)
class Model:
    opset_imports: tuple = field(default=())
    nodes: tuple = field(default=())

    def opset(self, domain=DEFAULT_DOMAIN):
        """Return the declared opset version for ``domain`` or ``None``."""
        for declared_domain, version in self.opset_imports:
            if declared_domain == normalize_domain(domain):
                return version
        return None

    def operators(self):
        """Return the ordered set of distinct ``(domain, op_type)`` pairs."""
        seen = []
        for node in self.nodes:
            key = (node.domain, node.op_type)
            if key not in seen:
                seen.append(key)
        return seen


def _parse_operator_set_id(data):
    domain = ""
    version = 0
    for field_number, wire_type, value in iter_fields(data):
        if field_number == 1 and wire_type == 2:
            domain = _text(value)
        elif field_number == 2 and wire_type == 0:
            version = value
    return normalize_domain(domain), version


def _parse_node(data):
    op_type = ""
    domain = DEFAULT_DOMAIN
    name = ""
    inputs = []
    outputs = []
    for field_number, wire_type, value in iter_fields(data):
        if field_number == 1 and wire_type == 2:
            inputs.append(_text(value))
        elif field_number == 2 and wire_type == 2:
            outputs.append(_text(value))
        elif field_number == 3 and wire_type == 2:
            name = _text(value)
        elif field_number == 4 and wire_type == 2:
            op_type = _text(value)
        elif field_number == 7 and wire_type == 2:
            domain = normalize_domain(_text(value))
    return Node(op_type=op_type, domain=domain, name=name,
                inputs=tuple(inputs), outputs=tuple(outputs))


def _parse_graph(data):
    nodes = []
    for field_number, wire_type, value in iter_fields(data):
        if field_number == 1 and wire_type == 2:
            nodes.append(_parse_node(value))
    return nodes


def parse_model(data):
    """Parse ONNX ``ModelProto`` bytes into a :class:`Model`."""
    opset_imports = []
    nodes = []
    for field_number, wire_type, value in iter_fields(data):
        if field_number == 8 and wire_type == 2:
            opset_imports.append(_parse_operator_set_id(value))
        elif field_number == 7 and wire_type == 2:
            nodes = _parse_graph(value)
    return Model(opset_imports=tuple(opset_imports), nodes=tuple(nodes))


def parse_model_file(path):
    with open(path, "rb") as handle:
        return parse_model(handle.read())
