# Catalog admission

`market_catalog.py` audits a separately versioned catalog against an explicit
policy. It requires a minimum greater than 16, minima for all four asset classes,
and declared instrument quotas. Seventeen is a synthetic test configuration,
not an agreed production catalog size. Quota shortfalls and invalid entries stay
visible; the tool does not drop a missing capability to make a form pass.

Each entry binds an exact row in a checksummed JSON-array listing snapshot, with
identity/units, declared asset and instrument labels, identity-validity window,
known/observed/retrieval clocks and option/prediction terms where applicable.
Duplicate IDs/contracts, unresolved identities, changed listing content,
expired terms and cutoff-ineligible knowledge fail admission. The policy must
explicitly permit synthetic snapshots; otherwise those entries cannot count.
The artifact retains synthetic provenance and never grants effect authority.

This bounded schema is an offline contract experiment. It does not implement a
normalizer for each provider's native catalog, verify asset/low-cap labels,
authenticate a source or establish independent snapshot custody. Snapshot hashes
and row equality establish internal consistency. A writer can still replace and
rehash the whole input. Contract-specific details beyond these fields need an
owning adapter before real options/prediction-market use. The eight controls use
synthetic listings only. There is no actual 17-market catalog or broader live
provider coverage claim. Real listing collection, source ownership, target count
and per-stratum quotas remain open decisions.

Call `audit_catalog(catalog, policy)` to get counts, invalid rows, coverage gaps,
input hashes and readiness. Keep the policy frozen before sampling. Historical
case/generator formats are unchanged; rollback stops using the additive audit.
