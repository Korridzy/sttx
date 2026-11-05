# Known Gaps

Qualification reduces regression risk for the declared speech backend, but it
does not establish equivalence for every input, model, or execution environment.

## Coverage

- The qualification envelope uses a finite English fixture, silence-prefix
  cases, and synthetic long-probe coverage. It is not a multilingual corpus or
  an exhaustive numerical equivalence proof.
- Timing comparisons allow bounded numerical drift. Changes within the allowed
  tolerances, or outside the observed cases, can go undetected.
- Exact direct dependency pins do not lock transitive packages, OS libraries,
  `ffmpeg`, hardware, or every numerical kernel. A pass in one environment is
  not proof for all supported Python versions and hosts.

## Model assets and overrides

- The default Parakeet runtime checks readability and nonempty files without
  rehashing large assets on every invocation. Subsequent cache tampering that
  leaves those checks satisfied is not automatically detected or repaired.
- Arbitrary explicit model bundles and user dependency overrides are outside
  the qualification guarantee.
- Silero cache repair requires network availability and upstream bytes matching
  the expected digest.

## Evidence boundaries

- Unit tests using fakes cannot establish the numerical behavior of native
  inference packages.
- Local qualification does not by itself prove hosted execution or publication.
- Filesystem setup and I/O failures can prevent evidence creation. Successful
  human-readable output alone is not proof that a qualification gate passed.
