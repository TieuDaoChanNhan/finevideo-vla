#!/usr/bin/env python3
"""
Dependency-free TFRecord + protobuf wire-format explorer.

No tensorflow/protobuf available in any project env (checked env_tools,
env_pose, finevideo-vla/env_motion_final -- none have it, and we don't
ad-hoc pip install per project convention). TFRecord framing itself is
simple enough to decode with stdlib struct/zlib; the payload (a serialized
tf.Example or tf.SequenceExample) is decoded with a generic schema-less
protobuf wire-format walker -- doesn't know field *names* a priori, but
recursively finds embedded messages and prints any length-delimited value
that decodes as valid UTF-8, which is enough to recover feature-map key
names and string-valued features (exactly what we need to answer "does
this tfrecord actually contain per-step video frames").

Usage:
    python3 data_prep/robovqa/inspect_tfrecord.py <path-to-tfrecord-file> [--record N]
"""
import struct
import sys
import zlib


def read_tfrecords(path):
    """Yield raw serialized-Example bytes for each record in a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                return
            length, = struct.unpack("<Q", header[:8])
            masked_crc, = struct.unpack("<I", header[8:12])
            data = f.read(length)
            footer = f.read(4)
            yield data


def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def walk_protobuf(buf, indent=0, max_depth=6):
    """Schema-less recursive walk of a protobuf-encoded byte string.
    Prints field number/wire-type, and for length-delimited fields either
    the decoded UTF-8 string (if printable) or recurses as a submessage.
    """
    pos = 0
    n = len(buf)
    prefix = "  " * indent
    while pos < n:
        try:
            tag, pos = read_varint(buf, pos)
        except IndexError:
            break
        field_num = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:  # varint
            val, pos = read_varint(buf, pos)
            print(f"{prefix}field={field_num} varint={val}")
        elif wire_type == 1:  # 64-bit
            val = buf[pos:pos + 8]
            pos += 8
            print(f"{prefix}field={field_num} fixed64={val.hex()}")
        elif wire_type == 5:  # 32-bit
            val = buf[pos:pos + 4]
            pos += 4
            print(f"{prefix}field={field_num} fixed32={val.hex()}")
        elif wire_type == 2:  # length-delimited: string, bytes, or embedded message
            length, pos = read_varint(buf, pos)
            val = buf[pos:pos + length]
            pos += length
            try:
                s = val.decode("utf-8")
                if s.isprintable() and len(s) < 300:
                    print(f"{prefix}field={field_num} str=\"{s}\"")
                    continue
            except UnicodeDecodeError:
                pass
            if max_depth > 0:
                print(f"{prefix}field={field_num} submessage (len={length}):")
                walk_protobuf(val, indent + 1, max_depth - 1)
            else:
                print(f"{prefix}field={field_num} bytes (len={length}, max depth reached)")
        else:
            print(f"{prefix}field={field_num} unknown wire_type={wire_type}, stopping")
            break


def main():
    path = sys.argv[1]
    record_idx = 0
    if "--record" in sys.argv:
        record_idx = int(sys.argv[sys.argv.index("--record") + 1])

    for i, data in enumerate(read_tfrecords(path)):
        if i < record_idx:
            continue
        print(f"=== record {i}, {len(data)} bytes ===")
        walk_protobuf(data, max_depth=8)
        break
    else:
        print(f"fewer than {record_idx + 1} records in file")


if __name__ == "__main__":
    main()
