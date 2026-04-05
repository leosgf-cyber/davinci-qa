#!/usr/bin/env python3
"""
Daily Audiovisual Q&A Generator
Generates 50 daily prompts across 4 categories and saves an interactive HTML file.
Runs via GitHub Actions — no local Mac required.
"""

import anthropic
import json
import os
import re
import sys
import requests
from datetime import date
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
HISTORY = ROOT / "history.json"
ARCHIVE = ROOT / "archive"
OUTPUT  = ROOT / "index.html"

ARCHIVE.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL             = "claude-sonnet-4-6"
MAX_HISTORY       = 500
QUESTIONS_PER_DAY = 50
DIFFICULTY_DIST   = "30% Beginner, 50% Intermediate, 20% Advanced"

# 4 categories with exact daily counts (total = 50)
CATEGORIES = {
    "A": {"label": "Tech — Software", "count": 8},
    "B": {"label": "Tech — Gear",     "count": 7},
    "C": {"label": "Creative",        "count": 18},
    "D": {"label": "Content",         "count": 17},
}

# Badge colors per type
TYPE_COLOR = {
    "Tech — Software": ("#0a2030", "#4FC3F7"),   # cyan
    "Tech — Gear":     ("#2a2010", "#FFD54F"),   # gold
    "Creative":        ("#2a1a2a", "#CE93D8"),   # purple
    "Content":         ("#1a2a1a", "#A5D6A7"),   # green
}

LEO_PROFILE = """
USER PROFILE — use this to make prompts more relevant and personally resonant:
- Brazilian videomaker and video editor living in the USA, 41 years old
- Strong Hollywood references AND Brazilian cinema (Cinema Novo, Glauber Rocha, Fernando Meirelles, Walter Salles, etc.)
- Deep knowledge of music theory and functional harmony
- Music taste: Rock, MPB, Bossa Nova, Pagode — references like Caetano Veloso, Gilberto Gil, Cazuza, Legião Urbana are welcome
- Can engage with politics and AI topics at an informed level
- Creative and Content prompts should occasionally draw on Brazilian cultural references, bilingual perspectives, or the immigrant/expat experience when relevant — but don't force it every time
"""

REDDIT_SUBREDDITS = [
    "videography", "photography", "Filmmakers",
    "travel", "solotravel", "personalfinance",
    "food", "gadgets",
]

# ── Reddit context ─────────────────────────────────────────────────────────────
def fetch_reddit_context() -> str:
    """Fetch top weekly posts from relevant subreddits as trending context."""
    lines = []
    headers = {"User-Agent": "qabot-daily/1.0"}
    for sub in REDDIT_SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/top.json?t=week&limit=8"
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue
            posts = resp.json().get("data", {}).get("children", [])
            for post in posts:
                title = post.get("data", {}).get("title", "").strip()
                if title:
                    lines.append(f"[r/{sub}] {title}")
        except Exception:
            continue  # skip flaky subreddit, never crash
    if lines:
        return "CURRENT TRENDING CONTEXT (week of {}):\n{}".format(
            date.today().isoformat(), "\n".join(lines)
        )
    return ""

# ── History ───────────────────────────────────────────────────────────────────
def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    try:
        return json.loads(HISTORY.read_text())
    except Exception:
        return []

def save_history(history: list[dict], new_questions: list[dict]):
    combined = history + [
        {"date": str(date.today()), "question": q["question"]}
        for q in new_questions
    ]
    trimmed = combined[-MAX_HISTORY:]
    HISTORY.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False))

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(cat_key: str, count: int, today_str: str,
                  reddit_ctx: str, recent_str: str) -> str:
    label = CATEGORIES[cat_key]["label"]

    if cat_key == "A":
        focus = f"""Generate exactly {count} prompts in the category "Tech — Software".
These are practical questions a videographer or content creator would ask about editing software.

TOPICS TO COVER (mix these):
- DaVinci Resolve: troubleshooting, effects, node workflow, Fusion, Fairlight, Deliver page
- Adobe Premiere Pro: timeline, effects, multicam, export
- After Effects: motion graphics, keyframes, expressions, plugins
- CapCut (desktop and mobile): effects, templates, transitions, trending features
- General editing concepts that apply across software (codec, bitrate, color space, proxy workflow)

QUESTION TYPES (mix): How-to, Why, Troubleshoot, What's the difference, Best practice, Keyboard shortcut"""

    elif cat_key == "B":
        focus = f"""Generate exactly {count} prompts in the category "Tech — Gear".
These are practical questions about physical camera gear and computing hardware.

TOPICS TO COVER (mix these):
- Camera bodies: settings, exposure, ISO, shutter speed, aperture, autofocus behavior
- Lenses: focal length choice, aperture, when to use wide/tele/prime, depth of field
- Shooting scenarios: weddings, travel, street, portraits, video run-and-gun
- PC/Mac optimization for video editing: RAM, GPU, storage, software settings
- Gear recommendations with CURRENT market prices

CRITICAL RULE FOR GEAR RECOMMENDATIONS:
- Today's date is {today_str}. Prices must reflect the current market.
- Always state clearly: "(new)" or "(used)" next to every price.
- Do NOT cite prices older than 6 months. If unsure, give a price range and say "check current listings".
- Budget recommendations should name specific current models (e.g. Sony ZV-E10 II, not ZV-E10 original if the II is now the main model at that price point).

QUESTION TYPES (mix): How-to, Which should I buy, How to set up, Troubleshoot, Best settings for X"""

    elif cat_key == "C":
        focus = f"""Generate exactly {count} prompts in the category "Creative".
These inspire and guide visual storytelling — for both photo and video creators.

TOPICS TO COVER (mix these):
- Cinematographer/director aesthetics: how to study and replicate Roger Deakins, Wong Kar-wai, Sofia Coppola, Agnès Varda, Emmanuel Lubezki, etc.
- Trending visual styles: what's popular RIGHT NOW on Instagram, TikTok, YouTube (use the Reddit context below as a guide)
- Vintage/film looks: Super 8, VHS, 35mm film grain, Kodachrome, expired film
- Color grading styles: split-toning, bleach bypass, day-for-night, muted vs saturated
- Inspiration resources: what to WATCH, READ, or LISTEN TO to become a better visual storyteller
- Location-based creativity: "What to shoot in Tokyo / Lisbon / New Orleans / NYC / small towns"
- Style mimicry: "How would you recreate the visual language of [specific film or creator]?"

{reddit_ctx}"""

    else:  # D
        focus = f"""Generate exactly {count} prompts in the category "Content".
These are content CREATION prompts — each one asks the model to produce a usable creative output.

FORMAT for each prompt: a specific, actionable request like:
- "Write a 60-second script for a vlog about [topic]"
- "Create a shot list for a [type] video"
- "Generate 5 Instagram caption ideas for a post about [topic]"
- "Write a YouTube video description for a [topic] video"
- "Create a week-long content calendar for a [niche] creator"

THEMES TO COVER (use the Reddit context for timely subject ideas):
- Gastronomy: restaurant reviews, recipes, food travel, cooking process
- Finance: budgeting tips, gear investment, freelance pricing, saving money while traveling
- Travel: destination guides, packing lists, solo travel, budget vs luxury
- Equipment reviews: written reviews, "Is it worth it?" scripts, comparison content
- Lifestyle: morning routines, workspace setups, productivity for creators
- Trending topics: pull from the Reddit context below for 4–5 of these prompts

{reddit_ctx}"""

    return f"""{focus}

DIFFICULTY DISTRIBUTION: {DIFFICULTY_DIST}

AVOID repeating questions similar to these recent ones:{recent_str}

{LEO_PROFILE}

FOLLOW-UP REQUIREMENTS:
- Generate exactly 4 follow-up prompts per question
- Each follow-up must be a standalone, copyable prompt that deepens or expands the original question
- Vary the angle across the 4: one more technical, one more creative, one more practical, one more conceptual
- Max 30 words each — concise and direct
- No answers — just the follow-up prompts themselves

OUTPUT FORMAT — respond ONLY with a valid JSON array, no markdown fences, no explanation:
[
  {{
    "type": "{label}",
    "difficulty": "beginner|intermediate|advanced",
    "question": "...",
    "followups": ["follow-up 1", "follow-up 2", "follow-up 3", "follow-up 4"]
  }},
  ...
]"""

# ── API call ──────────────────────────────────────────────────────────────────
def _call_api(client: anthropic.Anthropic, prompt: str) -> list[dict]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Salvage complete objects before the parse error
        objects = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
        items = []
        for obj in objects:
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
        if items:
            print(f"  ⚠ JSON parse error — salvaged {len(items)} items")
        else:
            raise

    valid = []
    for q in items:
        if isinstance(q, dict) and "question" in q and "type" in q:
            raw_fu = q.get("followups", [])
            followups = [str(f).strip() for f in raw_fu[:4]] if isinstance(raw_fu, list) else []
            valid.append({
                "type":       str(q.get("type", "Content")).strip(),
                "difficulty": str(q.get("difficulty", "intermediate")).strip().lower(),
                "question":   str(q["question"]).strip(),
                "followups":  followups,
            })
    return valid

# ── Orchestrator ──────────────────────────────────────────────────────────────
def generate_questions(client: anthropic.Anthropic,
                       reddit_ctx: str,
                       recent: list[str]) -> list[dict]:
    recent_str = ""
    if recent:
        sample = recent[-60:]
        recent_str = "\n" + "\n".join(f"- {q}" for q in sample)

    all_questions = []
    for key, cfg in CATEGORIES.items():
        print(f"  → Batch {key} ({cfg['label']}, {cfg['count']} prompts)...")
        prompt = _build_prompt(key, cfg["count"], date.today().isoformat(),
                               reddit_ctx, recent_str)
        batch = _call_api(client, prompt)
        print(f"     {len(batch)} returned")
        all_questions.extend(batch)

    if len(all_questions) < QUESTIONS_PER_DAY * 0.9:
        print(f"⚠ Warning: only {len(all_questions)} total prompts (expected {QUESTIONS_PER_DAY})")

    return all_questions

# ── HTML renderer ─────────────────────────────────────────────────────────────
def render_html(questions: list[dict], today: date) -> str:
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    total     = len(questions)
    types     = ["Tech — Software", "Tech — Gear", "Creative", "Content"]
    diffs     = ["beginner", "intermediate", "advanced"]

    counts_diff = {d: sum(1 for q in questions if q["difficulty"] == d) for d in diffs}
    counts_type = {t: sum(1 for q in questions if q["type"] == t) for t in types}

    type_buttons = "\n".join(
        f'<button class="filter-btn type-btn" data-type="{t}" onclick="filterType(this)">{t}</button>'
        for t in types
    )
    diff_buttons = "\n".join(
        f'<button class="filter-btn diff-btn" data-diff="{d}" onclick="filterDiff(this)">{d.capitalize()}</button>'
        for d in diffs
    )

    cards = []
    for i, q in enumerate(questions, 1):
        qtype = q["type"]
        diff  = q["difficulty"]
        bg, color = TYPE_COLOR.get(qtype, ("#1a1d27", "#aaa"))
        type_esc = qtype.replace('"', '&quot;')
        q_esc = q["question"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

        fu_cards = ""
        for fu in q.get("followups", []):
            fu_esc = fu.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            fu_cards += f'''  <div class="followup-card">
    <p class="followup-text" data-raw="{fu_esc}">{fu_esc}</p>
    <button class="copy-fu-btn" onclick="copyFU(this)" title="Copy follow-up">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>Copy
    </button>
  </div>\n'''

        cards.append(f'''<div class="card" data-type="{type_esc}" data-diff="{diff}">
  <div class="card-header">
    <div class="card-meta">
      <span class="q-num">Q{i:02d}</span>
      <span class="type-badge" style="background:{bg};color:{color};border-color:{color}">{qtype}</span>
    </div>
    <button class="copy-btn" onclick="copyQ(this)" title="Copy prompt">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>Copy
    </button>
  </div>
  <p class="q-text" data-raw="{q_esc}">{q_esc}</p>
  <button class="show-fu-btn" onclick="toggleFollowups(this)">
    <svg class="chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
    Show Follow-ups
  </button>
  <div class="followups-box">
    <div class="followups-label">Follow-up Prompts</div>
{fu_cards}  </div>
</div>''')

    cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Audiovisual Q&A — {today_iso}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      #0F1117; --surface: #1A1D27; --border: #2E3147;
    --accent:  #7C6AF7; --accent2: #4FC3F7; --text:   #E8EAF6;
    --muted:   #8B91B0; --gold:    #FFD54F; --radius: 10px;
  }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; min-height:100vh; padding-bottom:60px; }}
  header {{ background:linear-gradient(135deg,#1A0A3A 0%,#0F1117 60%); border-bottom:3px solid var(--accent); padding:32px 24px 24px; text-align:center; }}
  header .icon {{ font-size:32px; margin-bottom:8px; }}
  header h1 {{ font-size:clamp(1.4rem,3vw,2rem); font-weight:700; color:#fff; letter-spacing:-.5px; }}
  header .subtitle {{ color:var(--accent2); font-size:.95rem; margin-top:4px; }}
  header .meta {{ display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-top:16px; }}
  .stat {{ display:flex; flex-direction:column; align-items:center; background:rgba(124,106,247,.12); border:1px solid rgba(124,106,247,.3); border-radius:8px; padding:8px 14px; min-width:70px; }}
  .stat-num {{ font-size:1.3rem; font-weight:700; color:var(--accent); line-height:1; }}
  .stat-label {{ font-size:.68rem; color:var(--muted); margin-top:2px; text-transform:uppercase; letter-spacing:.5px; }}
  .controls {{ max-width:1100px; margin:24px auto 0; padding:0 16px; }}
  .search-wrap {{ position:relative; margin-bottom:16px; }}
  .search-wrap svg {{ position:absolute; left:12px; top:50%; transform:translateY(-50%); color:var(--muted); pointer-events:none; }}
  #search {{ width:100%; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); color:var(--text); font-size:.95rem; padding:10px 12px 10px 40px; outline:none; transition:border-color .2s; }}
  #search:focus {{ border-color:var(--accent); }}
  #search::placeholder {{ color:var(--muted); }}
  .filter-group {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:10px; }}
  .filter-label {{ font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; white-space:nowrap; min-width:80px; }}
  .filter-btn {{ background:var(--surface); border:1px solid var(--border); border-radius:20px; color:var(--muted); cursor:pointer; font-size:.8rem; padding:5px 12px; transition:all .18s; white-space:nowrap; }}
  .filter-btn:hover {{ border-color:var(--accent); color:var(--text); }}
  .filter-btn.active {{ background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }}
  .results-bar {{ font-size:.82rem; color:var(--muted); margin:10px 0 0; padding:0 2px; }}
  .results-bar span {{ color:var(--accent2); font-weight:600; }}
  .grid {{ max-width:1100px; margin:20px auto 0; padding:0 16px; display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:14px 16px; transition:border-color .2s,transform .15s; display:flex; flex-direction:column; gap:10px; }}
  .card:hover {{ border-color:var(--accent); transform:translateY(-2px); }}
  .card.hidden {{ display:none; }}
  .card-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
  .card-meta {{ display:flex; align-items:center; gap:6px; flex-wrap:wrap; }}
  .q-num {{ font-size:.75rem; font-weight:700; color:var(--accent); font-family:monospace; letter-spacing:.5px; }}
  .type-badge {{ border:1px solid; border-radius:4px; font-size:.7rem; font-weight:600; padding:2px 8px; text-transform:uppercase; letter-spacing:.4px; }}
  .copy-btn {{ background:transparent; border:1px solid var(--border); border-radius:6px; color:var(--muted); cursor:pointer; display:flex; align-items:center; gap:4px; font-size:.75rem; padding:4px 8px; transition:all .18s; white-space:nowrap; flex-shrink:0; }}
  .copy-btn:hover {{ border-color:var(--accent2); color:var(--accent2); }}
  .copy-btn.copied {{ border-color:#4CAF50; color:#4CAF50; }}
  .q-text {{ color:var(--text); font-size:.93rem; line-height:1.55; }}
  .show-fu-btn {{ background:transparent; border:1px solid var(--border); border-radius:6px; color:var(--muted); cursor:pointer; display:flex; align-items:center; gap:5px; font-size:.78rem; padding:5px 10px; transition:all .18s; width:fit-content; }}
  .show-fu-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .show-fu-btn.open {{ border-color:var(--accent); color:var(--accent); }}
  .show-fu-btn.open .chevron {{ transform:rotate(180deg); }}
  .chevron {{ transition:transform .2s; }}
  .followups-box {{ display:none; flex-direction:column; gap:8px; margin-top:2px; }}
  .followups-box.open {{ display:flex; }}
  .followups-label {{ font-size:.68rem; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:.6px; }}
  .followup-card {{ background:rgba(79,195,247,.07); border:1px solid rgba(79,195,247,.2); border-radius:6px; padding:10px 12px; display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
  .followup-text {{ color:#c5c9e0; font-size:.86rem; line-height:1.55; flex:1; }}
  .copy-fu-btn {{ background:transparent; border:1px solid rgba(79,195,247,.3); border-radius:5px; color:var(--muted); cursor:pointer; display:flex; align-items:center; gap:3px; font-size:.72rem; padding:3px 7px; transition:all .18s; white-space:nowrap; flex-shrink:0; }}
  .copy-fu-btn:hover {{ border-color:var(--accent2); color:var(--accent2); }}
  .copy-fu-btn.copied {{ border-color:#4CAF50; color:#4CAF50; }}
  .empty {{ grid-column:1/-1; text-align:center; padding:60px 20px; color:var(--muted); }}
  .empty .emoji {{ font-size:2.5rem; margin-bottom:12px; }}
  footer {{ text-align:center; color:var(--muted); font-size:.78rem; margin-top:40px; padding:0 16px; line-height:1.6; }}
</style>
</head>
<body>
<header>
  <div class="icon">◈</div>
  <h1>Daily Audiovisual Q&amp;A</h1>
  <div class="subtitle">Question Bank — {today_str}</div>
  <div class="meta">
    <div class="stat"><span class="stat-num">{total}</span><span class="stat-label">Prompts</span></div>
    <div class="stat"><span class="stat-num">{counts_type.get("Tech — Software",0)}</span><span class="stat-label">Tech SW</span></div>
    <div class="stat"><span class="stat-num">{counts_type.get("Tech — Gear",0)}</span><span class="stat-label">Tech Gear</span></div>
    <div class="stat"><span class="stat-num">{counts_type.get("Creative",0)}</span><span class="stat-label">Creative</span></div>
    <div class="stat"><span class="stat-num">{counts_type.get("Content",0)}</span><span class="stat-label">Content</span></div>
    <div class="stat"><span class="stat-num">{counts_diff["beginner"]}</span><span class="stat-label">Beginner</span></div>
    <div class="stat"><span class="stat-num">{counts_diff["intermediate"]}</span><span class="stat-label">Intermed.</span></div>
    <div class="stat"><span class="stat-num">{counts_diff["advanced"]}</span><span class="stat-label">Advanced</span></div>
  </div>
</header>

<div class="controls">
  <div class="search-wrap">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    <input id="search" type="text" placeholder="Search prompts…" oninput="applyFilters()">
  </div>
  <div class="filter-group">
    <span class="filter-label">Type</span>
    <button class="filter-btn type-btn active" data-type="all" onclick="filterType(this)">All</button>
    {type_buttons}
  </div>
  <div class="filter-group">
    <span class="filter-label">Difficulty</span>
    <button class="filter-btn diff-btn active" data-diff="all" onclick="filterDiff(this)">All</button>
    {diff_buttons}
  </div>
  <div class="results-bar">Showing <span id="visible-count">{total}</span> of {total} prompts</div>
</div>

<div class="grid" id="grid">
{cards_html}
  <div class="empty hidden" id="empty-state">
    <div class="emoji">🔍</div>
    <div>No prompts match your filters.</div>
  </div>
</div>

<footer>Generated on {today_iso} · Daily Audiovisual Q&amp;A · {total} prompts across 4 categories</footer>

<script>
  let activeType = 'all';
  let activeDiff = 'all';

  function filterType(btn) {{
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeType = btn.dataset.type;
    applyFilters();
  }}
  function filterDiff(btn) {{
    document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeDiff = btn.dataset.diff;
    applyFilters();
  }}
  function applyFilters() {{
    const query = document.getElementById('search').value.toLowerCase().trim();
    let visible = 0;
    document.querySelectorAll('.card').forEach(card => {{
      const mt = activeType === 'all' || card.dataset.type === activeType;
      const md = activeDiff === 'all' || card.dataset.diff === activeDiff;
      const mq = !query || card.querySelector('.q-text').textContent.toLowerCase().includes(query);
      if (mt && md && mq) {{ card.classList.remove('hidden'); visible++; }}
      else card.classList.add('hidden');
    }});
    document.getElementById('visible-count').textContent = visible;
    document.getElementById('empty-state').classList.toggle('hidden', visible > 0);
  }}
  function toggleFollowups(btn) {{
    const box  = btn.closest('.card').querySelector('.followups-box');
    const open = box.classList.toggle('open');
    btn.classList.toggle('open', open);
    btn.innerHTML = open
      ? `<svg class="chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="transform:rotate(180deg)"><polyline points="6 9 12 15 18 9"/></svg> Hide Follow-ups`
      : `<svg class="chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg> Show Follow-ups`;
  }}
  function copyFU(btn) {{
    const text = btn.closest('.followup-card').querySelector('.followup-text').dataset.raw;
    navigator.clipboard.writeText(text).then(() => {{
      btn.classList.add('copied');
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> ✓`;
      setTimeout(() => {{
        btn.classList.remove('copied');
        btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy`;
      }}, 1800);
    }}).catch(() => {{
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
      btn.classList.add('copied'); btn.textContent = '✓';
      setTimeout(() => {{ btn.classList.remove('copied'); btn.textContent = 'Copy'; }}, 1800);
    }});
  }}
  function copyQ(btn) {{
    const text = btn.closest('.card').querySelector('.q-text').dataset.raw;
    navigator.clipboard.writeText(text).then(() => {{
      btn.classList.add('copied');
      btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied`;
      setTimeout(() => {{
        btn.classList.remove('copied');
        btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
      }}, 2000);
    }}).catch(() => {{
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
      btn.classList.add('copied'); btn.textContent = '✓ Copied';
      setTimeout(() => {{
        btn.classList.remove('copied');
        btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
      }}, 2000);
    }});
  }}
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    today = date.today()
    print(f"[{today}] Generating {QUESTIONS_PER_DAY} prompts...")

    history = load_history()
    recent  = [h["question"] for h in history]

    print("Fetching Reddit context...")
    reddit_ctx = fetch_reddit_context()
    post_count = reddit_ctx.count("\n") if reddit_ctx else 0
    print(f"  → {post_count} trending posts fetched")

    client    = anthropic.Anthropic(api_key=api_key)
    questions = generate_questions(client, reddit_ctx, recent)

    print(f"  → {len(questions)} total prompts generated")

    html = render_html(questions, today)

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"  → Written: {OUTPUT}")

    archive_path = ARCHIVE / f"{today.isoformat()}.html"
    archive_path.write_text(html, encoding="utf-8")
    print(f"  → Archived: {archive_path}")

    save_history(history, questions)
    print(f"  → History updated ({min(len(history)+len(questions), MAX_HISTORY)} entries)")

    print(f"\nDone. {len(questions)} prompts for {today.isoformat()}.")

if __name__ == "__main__":
    main()
