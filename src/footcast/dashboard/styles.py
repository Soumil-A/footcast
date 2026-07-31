# ruff: noqa: E501
"""Visual system for the FootCast Streamlit product."""

APP_CSS = """
<style>
:root {
  --fc-bg: #050914;
  --fc-panel: rgba(15, 23, 42, 0.72);
  --fc-panel-strong: rgba(17, 26, 48, 0.92);
  --fc-border: rgba(148, 163, 184, 0.16);
  --fc-text: #f8fafc;
  --fc-muted: #94a3b8;
  --fc-cyan: #22d3ee;
  --fc-violet: #8b5cf6;
  --fc-pink: #f472b6;
  --fc-green: #34d399;
  --fc-amber: #fbbf24;
  --fc-red: #fb7185;
}

.stApp {
  color: var(--fc-text);
  background:
    radial-gradient(circle at 78% 0%, rgba(79, 70, 229, 0.19), transparent 34rem),
    radial-gradient(circle at 38% 32%, rgba(6, 182, 212, 0.08), transparent 26rem),
    linear-gradient(145deg, #050914 0%, #080d1c 48%, #050816 100%);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}

.stMainBlockContainer {
  max-width: 1240px;
  padding-top: 4.75rem;
  padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
  background: rgba(7, 11, 24, 0.92);
  border-right: 1px solid var(--fc-border);
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.6rem;
}

[data-testid="stSidebar"] h2 {
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--fc-cyan);
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div {
  background: rgba(15, 23, 42, 0.8);
  border-color: rgba(148, 163, 184, 0.18);
}

.stButton > button[kind="primary"] {
  min-height: 3.05rem;
  border: 0;
  border-radius: 0.8rem;
  background: linear-gradient(100deg, #06b6d4 0%, #6366f1 52%, #8b5cf6 100%);
  box-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
  color: white;
  font-weight: 800;
  letter-spacing: 0.025em;
  transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 0 42px rgba(34, 211, 238, 0.32);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--fc-border);
  border-radius: 1.1rem;
  background: linear-gradient(145deg, rgba(15, 23, 42, 0.76), rgba(9, 14, 29, 0.82));
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
}

[data-testid="stAlert"] {
  border: 1px solid var(--fc-border);
  border-radius: 0.9rem;
  background: rgba(15, 23, 42, 0.72);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0.45rem;
  padding: 0.35rem;
  border: 1px solid var(--fc-border);
  border-radius: 0.95rem;
  background: rgba(8, 13, 28, 0.72);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  min-height: 2.8rem;
  padding: 0 1.15rem;
  border-radius: 0.7rem;
  color: var(--fc-muted);
  font-weight: 750;
}

[data-testid="stTabs"] [aria-selected="true"] {
  background: linear-gradient(110deg, rgba(6, 182, 212, 0.16), rgba(139, 92, 246, 0.18));
  color: var(--fc-text);
}

[data-testid="stMetric"] {
  padding: 0.9rem 1rem;
  border: 1px solid var(--fc-border);
  border-radius: 0.9rem;
  background: rgba(15, 23, 42, 0.66);
}

[data-testid="stExpander"] {
  border-color: var(--fc-border);
  border-radius: 1rem;
  background: rgba(15, 23, 42, 0.62);
}

.fc-hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 2rem;
  margin-bottom: 1.7rem;
}

.fc-eyebrow {
  margin-bottom: 0.6rem;
  color: var(--fc-cyan);
  font-size: 0.76rem;
  font-weight: 850;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

.fc-title {
  margin: 0;
  font-size: clamp(2.7rem, 6vw, 5.3rem);
  font-weight: 900;
  letter-spacing: -0.065em;
  line-height: 0.92;
  background: linear-gradient(105deg, #ffffff 12%, #bae6fd 45%, #c4b5fd 85%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.fc-subtitle {
  max-width: 44rem;
  margin: 1rem 0 0;
  color: #a8b3c7;
  font-size: 1.05rem;
  line-height: 1.65;
}

.fc-live {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-top: 0.45rem;
  padding: 0.62rem 0.85rem;
  border: 1px solid rgba(52, 211, 153, 0.24);
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.08);
  color: #a7f3d0;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.fc-live-dot {
  width: 0.52rem;
  height: 0.52rem;
  border-radius: 50%;
  background: var(--fc-green);
  box-shadow: 0 0 14px rgba(52, 211, 153, 0.9);
}

.fc-matchup {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1.5rem;
  margin: 0.75rem 0 1.3rem;
  padding: 1.65rem clamp(1rem, 4vw, 2.4rem);
  border: 1px solid var(--fc-border);
  border-radius: 1.25rem;
  background:
    linear-gradient(120deg, rgba(6, 182, 212, 0.09), transparent 42%),
    linear-gradient(240deg, rgba(139, 92, 246, 0.11), transparent 42%),
    var(--fc-panel);
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
  backdrop-filter: blur(16px);
}

.fc-team {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.fc-team-away {
  flex-direction: row-reverse;
  text-align: right;
}

.fc-team-orb {
  display: grid;
  place-items: center;
  width: 4.5rem;
  height: 4.5rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 1.25rem;
  background: linear-gradient(145deg, rgba(34, 211, 238, 0.24), rgba(99, 102, 241, 0.26));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16), 0 0 32px rgba(34, 211, 238, 0.12);
  font-size: 1.35rem;
  font-weight: 900;
}

.fc-team-away .fc-team-orb {
  background: linear-gradient(145deg, rgba(139, 92, 246, 0.28), rgba(244, 114, 182, 0.2));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16), 0 0 32px rgba(139, 92, 246, 0.14);
}

.fc-team-role {
  color: var(--fc-muted);
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

.fc-team-name {
  margin-top: 0.15rem;
  font-size: clamp(1.05rem, 2.5vw, 1.55rem);
  font-weight: 850;
  letter-spacing: -0.03em;
}

.fc-vs {
  display: grid;
  place-items: center;
  width: 3rem;
  height: 3rem;
  border: 1px solid var(--fc-border);
  border-radius: 50%;
  background: rgba(2, 6, 23, 0.72);
  color: var(--fc-muted);
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.08em;
}

.fc-section-label {
  margin: 2.15rem 0 0.85rem;
  color: var(--fc-muted);
  font-size: 0.72rem;
  font-weight: 850;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.fc-prob-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
}

.fc-prob-card {
  padding: 1.15rem 1.2rem;
  border: 1px solid var(--fc-border);
  border-radius: 1rem;
  background: var(--fc-panel);
}

.fc-prob-card.is-leading {
  border-color: rgba(34, 211, 238, 0.5);
  background: linear-gradient(145deg, rgba(6, 182, 212, 0.15), rgba(15, 23, 42, 0.86));
  box-shadow: 0 0 36px rgba(34, 211, 238, 0.09);
}

.fc-prob-label {
  color: var(--fc-muted);
  font-size: 0.68rem;
  font-weight: 850;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.fc-prob-value {
  display: block;
  margin-top: 0.35rem;
  color: var(--fc-text);
  font-size: clamp(1.9rem, 3.8vw, 2.7rem);
  font-weight: 900;
  letter-spacing: -0.055em;
  line-height: 1;
  white-space: nowrap;
}

.fc-prob-track {
  display: flex;
  height: 0.72rem;
  margin: 1rem 0 0.5rem;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.1);
}

.fc-prob-track span:nth-child(1) { background: linear-gradient(90deg, #0891b2, #22d3ee); }
.fc-prob-track span:nth-child(2) { background: linear-gradient(90deg, #64748b, #94a3b8); }
.fc-prob-track span:nth-child(3) { background: linear-gradient(90deg, #7c3aed, #a78bfa); }

.fc-forecast-note {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.9rem;
  padding: 0.85rem 1rem;
  border: 1px solid var(--fc-border);
  border-radius: 0.9rem;
  background: rgba(15, 23, 42, 0.58);
  color: #cbd5e1;
  font-size: 0.88rem;
}

.fc-forecast-note strong { color: var(--fc-cyan); }
.fc-forecast-note span:last-child { color: var(--fc-muted); font-size: 0.76rem; }

.fc-empty {
  padding: 1.3rem 1.35rem;
  border: 1px dashed rgba(34, 211, 238, 0.27);
  border-radius: 1rem;
  background: rgba(6, 182, 212, 0.045);
}

.fc-empty strong { display: block; margin-bottom: 0.25rem; color: #cffafe; }
.fc-empty span { color: var(--fc-muted); font-size: 0.88rem; }

.fc-elo-grid {
  display: grid;
  grid-template-columns: 1fr minmax(12rem, 1.2fr) 1fr;
  gap: 0.9rem;
  align-items: stretch;
}

.fc-stat-card {
  padding: 1.15rem 1.2rem;
  border: 1px solid var(--fc-border);
  border-radius: 1rem;
  background: var(--fc-panel);
}

.fc-stat-card.right { text-align: right; }
.fc-stat-label { color: var(--fc-muted); font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.11em; }
.fc-stat-value { margin-top: 0.3rem; font-size: 2rem; font-weight: 900; letter-spacing: -0.04em; }
.fc-elo-center { text-align: center; }
.fc-elo-delta { color: var(--fc-cyan); font-size: 1.2rem; font-weight: 850; }
.fc-elo-track { position: relative; height: 0.48rem; margin: 0.75rem 0 0.45rem; border-radius: 999px; background: linear-gradient(90deg, #22d3ee, #334155 50%, #8b5cf6); }
.fc-elo-marker { position: absolute; top: 50%; width: 0.85rem; height: 0.85rem; transform: translate(-50%, -50%); border: 2px solid white; border-radius: 50%; background: #0f172a; box-shadow: 0 0 12px rgba(255,255,255,.35); }
.fc-elo-caption { color: var(--fc-muted); font-size: 0.72rem; }

.fc-form-pills {
  display: flex;
  gap: 0.45rem;
  margin: 0.75rem 0 1rem;
}

.fc-form-pill {
  display: grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  border-radius: 0.58rem;
  font-size: 0.72rem;
  font-weight: 900;
}

.fc-form-pill.win { color: #a7f3d0; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.24); }
.fc-form-pill.draw { color: #fde68a; background: rgba(245, 158, 11, 0.13); border: 1px solid rgba(251, 191, 36, 0.22); }
.fc-form-pill.loss { color: #fecdd3; background: rgba(244, 63, 94, 0.13); border: 1px solid rgba(251, 113, 133, 0.22); }

.fc-mini-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.55rem;
  margin-bottom: 1rem;
}

.fc-mini-stat { padding: 0.72rem; border-radius: 0.72rem; background: rgba(2, 6, 23, 0.35); }
.fc-mini-stat span { display: block; color: var(--fc-muted); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; }
.fc-mini-stat strong { display: block; margin-top: 0.15rem; font-size: 1.15rem; }

.fc-history-list { display: grid; gap: 0.5rem; }
.fc-history-row { display: grid; grid-template-columns: 3.1rem minmax(0, 1fr) auto; align-items: center; gap: 0.65rem; padding: 0.65rem 0.72rem; border-radius: 0.72rem; background: rgba(2, 6, 23, 0.28); }
.fc-venue { color: var(--fc-muted); font-size: 0.67rem; font-weight: 800; letter-spacing: 0.08em; }
.fc-opponent { overflow: hidden; font-size: 0.82rem; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
.fc-score { font-variant-numeric: tabular-nums; font-size: 0.82rem; font-weight: 900; }
.fc-score.win { color: var(--fc-green); }
.fc-score.draw { color: var(--fc-amber); }
.fc-score.loss { color: var(--fc-red); }

.fc-h2h-list { display: grid; gap: 0.6rem; }
.fc-h2h-row { display: grid; grid-template-columns: 6.2rem minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 0.8rem; padding: 0.82rem 0.9rem; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 0.8rem; background: rgba(2, 6, 23, 0.24); }
.fc-h2h-date { color: var(--fc-muted); font-size: 0.72rem; }
.fc-h2h-home { text-align: right; font-weight: 700; }
.fc-h2h-away { font-weight: 700; }
.fc-h2h-score { padding: 0.25rem 0.55rem; border-radius: 0.5rem; background: rgba(99, 102, 241, 0.14); font-weight: 900; font-variant-numeric: tabular-nums; }

.fc-sidebar-brand { margin-bottom: 1.45rem; }
.fc-sidebar-mark { display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; margin-bottom: 0.7rem; border-radius: 0.75rem; background: linear-gradient(145deg, #06b6d4, #7c3aed); box-shadow: 0 0 24px rgba(99,102,241,.28); font-weight: 900; }
.fc-sidebar-name { font-size: 1.05rem; font-weight: 900; letter-spacing: -0.03em; }
.fc-sidebar-caption { color: var(--fc-muted); font-size: 0.72rem; }

.fc-meta {
  display: grid;
  gap: 0.48rem;
  margin-top: 1rem;
  padding: 0.85rem;
  border: 1px solid var(--fc-border);
  border-radius: 0.8rem;
  background: rgba(15, 23, 42, 0.46);
}

.fc-meta-row { display: flex; justify-content: space-between; gap: 0.7rem; color: var(--fc-muted); font-size: 0.68rem; }
.fc-meta-row strong { overflow: hidden; color: #dbeafe; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }

@media (max-width: 760px) {
  .stMainBlockContainer { padding-top: 4.25rem; }
  .fc-hero { display: block; }
  .fc-live { margin-top: 1rem; }
  .fc-matchup { grid-template-columns: 1fr; text-align: center; }
  .fc-team, .fc-team-away { flex-direction: column; text-align: center; }
  .fc-vs { margin: -0.3rem auto; }
  .fc-prob-grid { grid-template-columns: 1fr; }
  .fc-elo-grid { grid-template-columns: 1fr; }
  .fc-stat-card, .fc-stat-card.right { text-align: center; }
  .fc-h2h-row { grid-template-columns: 1fr auto 1fr; }
  .fc-h2h-date { grid-column: 1 / -1; }
  .fc-forecast-note { align-items: flex-start; flex-direction: column; }
}
</style>
"""
