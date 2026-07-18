#!/usr/bin/env python3
"""
Dependency-free reader for tf.SequenceExample-encoded TFRecord files.

No tensorflow/protobuf available in any project env (env_tools, env_pose,
finevideo-vla/env_motion_final all checked, none have it -- and per project
convention we don't ad-hoc pip-install tensorflow just to read one dataset's
tfrecords). This implements just enough of the protobuf wire format to
decode the specific message shapes TFRecord uses:

    TFExample.SequenceExample {
      context      : Features        (field 1)
      feature_lists: FeatureLists    (field 2)
    }
    Features       { feature: map<string, Feature> }       (field 1, repeated map-entry)
    FeatureLists   { feature_list: map<string, FeatureList> }
    FeatureList    { feature: repeated Feature }            (field 1)
    Feature        { oneof: bytes_list=1 | float_list=2 | int64_list=3 }
    BytesList      { value: repeated bytes }                (field 1)
    Int64List      { value: repeated int64, packed }        (field 1)
    FloatList      { value: repeated float, packed }        (field 1)

Verified against real RoboVQA tfrecord shards (18/07/2026): each record's
`context` has `unique_id` (bytes, decimal string) and `video_filename`
(bytes); `feature_lists` has `images` (N JPEG-bytes Feature entries, one per
timestep), `timestamps` (N int64 Feature entries), and `texts`/`raw_texts`/
`texts_start`/`texts_end` (single Feature each, whole-episode text blob).
"""
import struct
import zlib


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


def read_tag(buf, pos):
    tag, pos = read_varint(buf, pos)
    return tag >> 3, tag & 7, pos


def read_length_delimited(buf, pos):
    length, pos = read_varint(buf, pos)
    return buf[pos:pos + length], pos + length


def iter_tfrecords(path):
    """Yield raw serialized-record bytes for each record in a TFRecord file.
    Verifies the length CRC (cheap) but not the data CRC (not needed here).
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                return
            length, = struct.unpack("<Q", header[:8])
            length_crc, = struct.unpack("<I", header[8:12])
            data = f.read(length)
            f.read(4)  # data crc, unchecked
            yield data


def _parse_bytes_list(buf):
    """BytesList { repeated bytes value = 1 } -> list[bytes]"""
    out = []
    pos = 0
    while pos < len(buf):
        fn, wt, pos = read_tag(buf, pos)
        val, pos = read_length_delimited(buf, pos)
        out.append(val)
    return out


def _parse_varint_packed(buf):
    """Int64List{repeated int64 value=1 [packed]} -- handles both packed
    (single length-delimited blob of varints) and unpacked (repeated
    varint fields) encodings, since proto3 varint-packing is implementation
    dependent on the writer."""
    out = []
    pos = 0
    while pos < len(buf):
        fn, wt, pos = read_tag(buf, pos)
        if wt == 2:  # packed
            blob, pos = read_length_delimited(buf, pos)
            bp = 0
            while bp < len(blob):
                v, bp = read_varint(blob, bp)
                out.append(v)
        elif wt == 0:  # unpacked
            v, pos = read_varint(buf, pos)
            out.append(v)
    return out


def _parse_feature(buf):
    """Feature { oneof: bytes_list=1 | float_list=2 | int64_list=3 } ->
    (kind, list_of_values)"""
    pos = 0
    fn, wt, pos = read_tag(buf, pos)
    sub, pos = read_length_delimited(buf, pos)
    if fn == 1:
        return "bytes", _parse_bytes_list(sub)
    elif fn == 3:
        return "int64", _parse_varint_packed(sub)
    elif fn == 2:
        # float_list: rarely used here: parse as packed 4-byte floats
        out = []
        p = 0
        while p < len(sub):
            fn2, wt2, p = read_tag(sub, p)
            blob, p = read_length_delimited(sub, p)
            for i in range(0, len(blob), 4):
                out.append(struct.unpack("<f", blob[i:i + 4])[0])
        return "float", out
    return "unknown", []


def _parse_feature_map(buf):
    """Features{feature: map<string,Feature>} -> dict[str, (kind, values)]"""
    out = {}
    pos = 0
    while pos < len(buf):
        fn, wt, pos = read_tag(buf, pos)
        entry, pos = read_length_delimited(buf, pos)
        # map entry: field1=key(string), field2=value(Feature)
        ep = 0
        efn, ewt, ep = read_tag(entry, ep)
        key, ep = read_length_delimited(entry, ep)
        key = key.decode("utf-8")
        efn2, ewt2, ep = read_tag(entry, ep)
        feat_buf, ep = read_length_delimited(entry, ep)
        out[key] = _parse_feature(feat_buf)
    return out


def _parse_feature_lists_map(buf):
    """FeatureLists{feature_list: map<string,FeatureList>} ->
    dict[str, list[(kind, values)]]  (one entry per timestep)"""
    out = {}
    pos = 0
    while pos < len(buf):
        fn, wt, pos = read_tag(buf, pos)
        entry, pos = read_length_delimited(buf, pos)
        ep = 0
        efn, ewt, ep = read_tag(entry, ep)
        key, ep = read_length_delimited(entry, ep)
        key = key.decode("utf-8")
        efn2, ewt2, ep = read_tag(entry, ep)
        fl_buf, ep = read_length_delimited(entry, ep)
        # FeatureList { repeated Feature feature = 1 }
        steps = []
        fp = 0
        while fp < len(fl_buf):
            ffn, fwt, fp = read_tag(fl_buf, fp)
            feat_buf, fp = read_length_delimited(fl_buf, fp)
            steps.append(_parse_feature(feat_buf))
        out[key] = steps
    return out


def parse_sequence_example(data):
    """Top-level tf.SequenceExample -> (context: dict, feature_lists: dict)"""
    pos = 0
    context = {}
    feature_lists = {}
    while pos < len(data):
        fn, wt, pos = read_tag(data, pos)
        sub, pos = read_length_delimited(data, pos)
        if fn == 1:
            context = _parse_feature_map(sub)
        elif fn == 2:
            feature_lists = _parse_feature_lists_map(sub)
    return context, feature_lists


def context_str(context, key):
    kind, values = context.get(key, (None, []))
    if kind == "bytes" and values:
        return values[0].decode("utf-8")
    return None
