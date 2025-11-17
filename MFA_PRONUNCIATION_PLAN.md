# Montreal Forced Alignment Migration Plan

## Goals
- Replace the Bournemouth Forced Aligner (BFA) dependency with Montreal Forced Aligner (MFA) so the `/pronunciation` route can use a production-grade, well-documented aligner.
- Preserve existing API contract (`AnalyzeResponse`) while improving reliability and observability of forced alignment.
- Keep the solution maintainable by isolating MFA-specific code and clearly documenting install/runtime assumptions.

## Steps
1. **Assess current pronunciation flow**
   - Map out how `routes_scripted.py` loads/normalizes audio, phonemizes text, and currently calls BFA.
   - Identify reused helpers (tokenization, scoring) that should remain untouched.
   - Note where temporary files are written and how request tracking and error handling work to ensure parity.

2. **Design MFA integration surface**
   - Decide whether to invoke MFA via CLI (`mfa align` / `mfa alignments`) or Python API (if available) to suit our request/response lifecycle.
   - Determine required acoustic/dictionary models and where to store/load them (e.g., shipped in `pretrained_models` or downloaded lazily).
   - Outline expected inputs (audio wav, transcription text) and outputs (word/phone timings) from MFA, and how to convert them into `WordPronunciation` + `PhonemeScore`.

3. **Implement MFA wrapper**
   - Write a reusable helper (e.g., `run_mfa_alignment()` in a new module under `main/`) that:
     - Writes audio to temp WAV, prepares a temporary corpus structure MFA expects.
     - Calls MFA (subprocess) with proper CLI flags and timeout handling.
     - Parses MFA TextGrid or JSON outputs into structured phoneme data with timestamps/confidence proxies.
     - Cleans up temporary folders and surfaces meaningful errors.

4. **Refactor `/pronunciation` route**
   - Remove BFA-specific imports, env handling, and scoring logic.
   - Integrate the new MFA helper, mapping MFA output phoneme durations into scores (e.g., coverage/presence) consistent with the existing UI expectations.
   - Preserve request tracking, logging, and fallback behaviors (text-only path, phoneme scoring when browser transcript equals text).

5. **Validation & documentation**
   - Add targeted unit/integration hooks (or local test instructions) showing how to run MFA alignment against a sample clip (e.g., `test_bus_class.py`).
   - Document prerequisites (`pip install montreal-forced-aligner`, download models) and configuration knobs in README or a dedicated section.
   - Smoke-test the `/pronunciation` endpoint locally via `http://localhost:3000/assignments/...` to confirm JSON matches expectations, logging new data structures.

