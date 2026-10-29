# Completed-study configuration examples

These files record the decisions used for the completed 2026 pilot studies.
They are examples and audit records, not reusable templates.

- `transcription_pilot_2026.json` contains the study-specific transcription
  conditions, normalization decisions, exclusions, and manual overrides.
- `mos_pilot_2026.json` contains the study-specific MOS conditions, incomplete
  session decisions, eligibility correction, and attention-check overrides.

The MOS pilot config also records the temporary legacy item-ID harmonization
used for that completed run. That compatibility rule is intentionally absent
from the reusable pipeline. Future MOS inputs must provide canonical unique
`item_id` values before analysis.

Use the sanitized templates under `config/` when starting a new study. If this
repository is published, review completed-study configs before sharing because
they contain pseudonymous participant and session identifiers.
