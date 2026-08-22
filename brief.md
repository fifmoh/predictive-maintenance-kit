---
company: bertrandt
repo: predictive-maintenance-kit
stack: [Python, researchops, XGBoost, PyTorch LSTM, Streamlit, Plotly]
timebox: "MVP: 1 weekend — one dataset, one model, one dashboard. Full: 4 weeks — second dataset + English technical note."
dataset: "NASA C-MAPSS turbofan degradation (+ AI4I 2020 as the transfer axis)"
status: demo-ready
---
# Predictive maintenance on sensor data: remaining-useful-life kit

## Why this project exists (the thinking)

The chain of reasoning: Bertrandt's Rabat site builds *vehicle-dynamics models* and runs *test & validation* — meaning their daily substrate is **sensor and simulation data**, and their clients' expensive question is "when does this unit fail / need service?" Predictive maintenance is not a buzzword there; it is the business case of half the industrial ML market. The sibling project ([[projects/mascir/brief|defect benchmark]]) covers the *image* side of industrial AI; this one covers the *time-series* side — together they are one kit family (cluster 4 of [[00-Gap-Map]]) attacking the same buyer-need from two data modalities, which is why they share a build.

The specific design choice that earns this project its place: it doesn't stop at the model. A degradation forecast is worthless to an engineering buyer until it becomes a **decision rule** — the alert threshold — and thresholds are *economic* decisions (early replacement wastes life; late replacement risks failure). Building the cost-curve turns a student demo into an engineering argument.

## What you actually learn by building it

- **Time-series ML the honest way**: degradation data is irregular, censored, and non-stationary; you'll learn feature engineering from raw sensor streams, why random CV splits leak, and how RUL framing differs from plain regression.
- **The economics layer**: translating model output into maintenance cost curves — the skill that makes engineers trust data people.
- **Transfer discipline**: running the same kit on a second dataset (AI4I) teaches what generalizes and what was dataset luck — a mini-lesson in external validity.

## Its real value beyond the CV

- **Genuinely useful artifact**: any small Moroccan industrial operation with sensor logs could run this kit on their data tomorrow — it's a starter analytics department in a repo.
- **The cost-threshold analysis is a reusable thinking tool**: once you've built one, you'll frame every future ML task as "model → decision → money," which is how senior engineers actually talk.
- **Honest ceiling**: public datasets, no real client data; the value is the demonstrated method and economics, not operational deployment. Say so plainly in interviews — that honesty itself lands well with German-engineering culture.

## Why <company> cares
[[companies/bertrandt]] briefing: sensor/test data is their substrate; QC and predictive maintenance are named client needs; the MVTec sibling covers the visual axis.

## MVP scope
- NASA C-MAPSS: RUL prediction (XGBoost + simple LSTM), Streamlit dashboard, English README.

## Full scope
- AI4I 2020 transfer run; cost-aware threshold analysis (early-vs-late replacement curves); 4-page English technical note; demo video.

## Demo script
1. Degradation curves. 2. RUL prediction context. 3. Change illustrative cost units and show the validation-selected threshold without claiming currency savings. 4. Transfer evidence: same decision contracts, different target/model.

## CV bullets
- Implemented deterministic FD001 engine splits, train-only preprocessing/windowing, leakage checks, and a validation-selected 27-cell illustrative threshold-cost surface.
- Implemented a fixed AI4I transfer-study split and a four-page English technical note; real NASA/UCI metrics and savings claims remain unpublished.
