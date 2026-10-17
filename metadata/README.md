# Study metadata

- `record_manifest.csv` lists the 71 GM and 40 centrifuge records described in
  the article, their study use, the magnitude input used for inference, and
  how the raw data may be accessed.
- `reference_labels.csv` preserves the available literature/reference labels,
  methods, studies, and notes. Some GM records have multiple candidate values
  in one source field; these are retained as supplied rather than collapsed.
- `fold_event_assignments.csv` records the event-grouped training/validation
  assignment for each of the five model folds. The same training records can
  be training data in four folds and out-of-fold validation data in one fold.

The raw acceleration histories are not included. In particular, records
obtained through personal communications are not redistributed because
redistribution permission was not obtained.

