# NeuroBrain Recovery

If all conversation history is lost, start here.

## Read first
1. `docs/CURRENT_STATE.md`
2. `docs/ARCHITECTURE.md`
3. `docs/HARDWARE.md`
4. `docs/MODELS.md`
5. `docs/INSTALL.md`
6. `docs/DESIGN_DECISIONS.md`

Then inspect source and Git history.

## Known checkpoints
- `fe85f97` — integrated voice + vision + GPIO.
- `a2f80fe` — independent live vision service.
- `49ffe01` — semantic action router.
- `32bd600` — GPIO17 READ/SET/TOGGLE.
- `8f87aa7` — stable voice pipeline.
- `4d3adbb` — physical recovery PDF.

Never restore an old commit over a working tree without preserving the current state.

## Start
```bash
/home/drpeker/neurobrain/venv/bin/python \
  /home/drpeker/neurobrain/start_neurobrain_vision_dev.py
```

## Smoke test
1. “What do you see?”
2. Point camera at a clear object and repeat.
3. “Turn on the LED.”
4. Ask an ordinary conversational question.

## Vision failure isolation
Test in order: `/dev/video0` → OpenCV → `vision_detect_working.py` → VisionState → VisionService → VisionRuntime → integrated NeuroBrain.

## Engineering rule
When a state works physically: stop → checkpoint → commit → push → experiment only in a new copy.


## Complete reconstruction map
See `docs/SYSTEM_DIAGRAMS.md` for the full end-to-end diagrams, model roles, action/vision flows, failed experiments and recovery map.
