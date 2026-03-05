---
source: https://github.com/lablab-ai/community-content/blob/main/tutorials/en/openclaw-tutorial-part-one-ai-hackathons.mdx
title: OpenCLAW Tutorial - Part One - AI Hackathons
fetched_at: 2026-02-22
---

# OpenCLAW Tutorial - Part One: AI Hackathons

## Quick Start in a Hackathon (30-minute setup)

### Step 1: Environment (5 min)

```bash
# Option A: GitHub Codespaces (recommended for hackathons)
# Click "Code" → "Create Codespace on main"

# Option B: Local
git clone https://github.com/lablab-ai/openclaw.git
cd openclaw
npm install
```

### Step 2: Choose template (3 min)

```bash
# See all available hackathon templates
npx openclaw list templates

# Most popular for hackathons:
npx openclaw create my-hackathon-app --template hackathon-starter
```

Common templates:

- `hackathon-starter`          — chat + 5 tools
- `multi-agent-debate`         — two agents arguing + judge
- `research-agent`             — web search + report generation
- `content-creation-pipeline`  — idea → script → image → video
- `personal-ai-assistant`      — calendar + email + todo

### Step 3: Customize (10–20 min)

Edit `agents/` and `tools/` folders

### Step 4: Run locally (2 min)

```bash
npm run dev
# Open http://localhost:5173
```

### Step 5: Deploy (5–10 min)

```bash
# Option 1: Cloudflare Pages
npm run deploy:cf

# Option 2: Vercel
vercel

# Option 3: Hugging Face Spaces
# (upload via web interface)
```
