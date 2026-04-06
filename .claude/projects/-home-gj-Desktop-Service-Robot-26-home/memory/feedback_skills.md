---
name: skills_layer_ownership
description: User writes all concrete skill implementations themselves; don't create skill wrapper files
type: feedback
---

Do not create or modify files in common/skills/ — the user handles all skill layer implementations.

**Why:** User already has hardware APIs in place (agv_api, head_control, camera, slide_control) and wants full control over skill abstractions.

**How to apply:** When building states/behaviors, reference skill functions as abstract imports with clear interfaces, but don't create the actual skill files. Focus on architecture (state machine, config, context, states, behaviors, main).
