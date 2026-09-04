# nlh-evidence

Empirical evidence repo for a $0 no-human-in-the-loop automation stack.

- K1 device flow, K2 sealed-box secrets, K3 TOTP session persistence,
  K4 headless DOM, K5 unattended CI secret consumption.
- The workflow in `.github/workflows/k5.yml` proves an Actions secret
  (fetched + rotated via REST, never stored as plaintext) reaches a
  scheduled/dispatched job unattended.
