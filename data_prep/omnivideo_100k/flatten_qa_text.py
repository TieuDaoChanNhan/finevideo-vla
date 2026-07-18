#!/usr/bin/env python3
"""
Flatten OmniVideo-100K's QA-only text stream (train_oe_70k.jsonl +
train_mcq_30k.jsonl) into Megatron-ready flat JSONL ({"text": ...} per line).

This is the "pure text, no video token" half of OmniVideo-100K -- see
data_prep/omnivideo_100k/flatten_video_captions.py for the other half
(video_path + segments -> seed2/cosmos/avclm + caption/speech, which depends
on Step A having been run on the extracted videos first).

Open-ended records -> "Q: <question>\nA: <answer>"
MCQ records        -> "Q: <question>\nA. ...\nB. ...\n...\nAnswer: <letter>"

`analysis.connections` (present on both) is appended as a reasoning hint --
it explains *why* the answer requires cross-modal (audio+visual) reasoning,
which is exactly the kind of signal DISCUSS-1 wants, so it's kept rather
than dropped.

Usage:
    python3 data_prep/omnivideo_100k/flatten_qa_text.py
"""
import json
import os
import string

SRC_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k"
OE_FILE = os.path.join(SRC_DIR, "train_oe_70k.jsonl")
MCQ_FILE = os.path.join(SRC_DIR, "train_mcq_30k.jsonl")
OUT_PATH = "/p/data1/mmlaion/shared/vla/omnivideo_100k_flat/omnivideo_100k_qa_flat.jsonl"


def format_oe(rec):
    answer = rec["answer"]
    # event_sequence_ordering (open-ended variant) gives answer as a list of
    # step labels, e.g. ["B", "C", "A"] -- render as an arrow chain instead
    # of Python's repr (which str.format would otherwise produce verbatim).
    if isinstance(answer, list):
        answer = " -> ".join(str(a) for a in answer)
    text = f"Q: {rec['question']}\nA: {answer}"
    reasoning = (rec.get("analysis") or {}).get("connections")
    if reasoning:
        text += f"\nReasoning: {reasoning}"
    return text


def format_mcq(rec):
    # Most tasks use question/options; event_sequence_ordering instead uses
    # question_textual/options_textual (+ question_indexed/options_indexed,
    # a numbered-event variant we don't need since textual is more natural
    # language). Both shapes carry a plain `answer` letter.
    question = rec.get("question") or rec.get("question_textual")
    options = rec.get("options") or rec.get("options_textual")
    lines = [f"Q: {question}"]
    for letter, option in zip(string.ascii_uppercase, options):
        lines.append(f"{letter}. {option}")
    lines.append(f"Answer: {rec['answer']}")
    reasoning = (rec.get("analysis") or {}).get("connections")
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n_oe = n_mcq = n_skipped = 0

    with open(OUT_PATH, "w") as out_f:
        with open(OE_FILE) as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("question") or not rec.get("answer"):
                    n_skipped += 1
                    continue
                out_f.write(json.dumps({"text": format_oe(rec)}, ensure_ascii=False) + "\n")
                n_oe += 1

        with open(MCQ_FILE) as f:
            for line in f:
                rec = json.loads(line)
                question = rec.get("question") or rec.get("question_textual")
                options = rec.get("options") or rec.get("options_textual")
                if not question or not options or not rec.get("answer"):
                    n_skipped += 1
                    continue
                out_f.write(json.dumps({"text": format_mcq(rec)}, ensure_ascii=False) + "\n")
                n_mcq += 1

    print(f"open-ended: {n_oe}, mcq: {n_mcq}, skipped: {n_skipped} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
