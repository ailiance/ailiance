"""ailiance Realtime API wrap — OpenAI Realtime-compatible WebSocket.

Sub-package of the ailiance gateway. Exposes a single WebSocket
endpoint at ``/v1/realtime`` whose event protocol mirrors OpenAI's
Realtime API, dispatching to Kyutai STT, LiteLLM, and voice-bridge
TTS under the hood.

Designed to be drop-in for any existing OpenAI Realtime client — set
``OPENAI_BASE_URL`` (or equivalent) to the ailiance gateway URL and
talk to ``/v1/realtime``.

See docs/specs/2026-05-24-ailiance-realtime-wrap.md (in the
le-mystere-professeur-zacus repo) for the v1 scope and event list.
"""
