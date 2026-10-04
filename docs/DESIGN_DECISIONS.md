# NeuroBrain Design Decisions

## Core over model
The LLM is replaceable. Identity, safety, hardware authorization, world state and action execution belong outside it.

## No direct LLM hardware control
Qwen produces semantic intent. Deterministic Python validates and executes physical actions.

## One resident Qwen
The Pi 4 has 4 GB RAM. The project rejected running two Qwen models simultaneously. The working design uses one resident Qwen and may call it twice for CHAT.

## Abandoned unified single-call experiment
A Qwen response+operations single-call design was rejected because:
- JSON sometimes had trailing invalid characters;
- prompt evaluation was slow;
- server-level `--json-schema` caused HTTP 400 / sampler initialization failure in the tested llama.cpp build;
- a compact line protocol produced contradictory operations and omitted requested conversation.

Do not revive this path without an isolated experiment and concrete reason.

## Vision is separate perception
Raw frames are not sent to Qwen. YOLO/NCNN detects objects continuously; Core consumes structured VisionService state.

## Temporal filtering
The working filter is history 5 / minimum hits 3. It can drop an object on a current-frame miss. This was accepted during fast camera movement. Do not casually retune it.

## Distance
`near/medium/far` is a bbox-area heuristic only. Autonomous safety requires physical range sensors.

## Performance
Correctness/recoverability precede latency optimization. Qwen routing is the current bottleneck. Optimize only in a new development copy while preserving `fe85f97`.
