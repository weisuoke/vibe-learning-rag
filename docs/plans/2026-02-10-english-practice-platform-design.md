# English Practice Platform for International Developers - Design Document

**Date**: 2026-02-10
**Status**: Design Phase
**Target Launch**: MVP in 6-7 weeks

---

## Executive Summary

An AI-powered English practice platform targeting international developers who want to improve their American accent for technical scenarios. The platform combines structured lessons focused on developer-specific scenarios (technical interviews, code reviews, team meetings) with instant AI-powered pronunciation feedback.

**Key Differentiators:**
- Specialized content for developers (not generic English learning)
- Instant pronunciation feedback (vs. $50-100/hour human tutors)
- Practice anytime without scheduling or embarrassment
- Measurable progress tracking

---

## 1. Product Overview & Core Value Proposition

### Product Name Ideas
- AccentDev
- DevSpeak
- CodeVoice
- TechTalk
- (To be finalized)

### One-Line Pitch
"Practice American accent for technical scenarios - get instant AI feedback on your pronunciation"

### Target User
**Primary**: International developers (non-native English speakers) who want to improve their American accent for:
- Technical interviews
- Code reviews and pair programming
- Team meetings and standups
- Conference talks and presentations

**Why This Audience:**
- ✅ Founder is part of this audience (deep understanding of pain points)
- ✅ Easy to reach through dev communities (Reddit, Twitter, Discord, GitHub)
- ✅ Clear ROI: better communication = career advancement, higher salaries, leadership roles
- ✅ Specific scenarios: technical interviews, code reviews, presentations, team meetings
- ✅ Can leverage founder's network for early users and feedback
- ✅ Already comfortable with AI tools (heavy users of Claude/ChatGPT)

### Core Value Proposition

1. **Specialized Content**: Not generic English - focused on dev scenarios with technical vocabulary
2. **Instant Feedback**: Record yourself, get immediate pronunciation analysis (vs. expensive human tutors)
3. **Practice Anytime**: No scheduling, no embarrassment, unlimited practice
4. **Measurable Progress**: Track improvement over time with pronunciation scores

### MVP Feature Set

**In Scope:**
- ✅ **Structured dev scenarios** (~10-20 pre-built lessons)
- ✅ **Pronunciation practice with instant feedback**
- ✅ Basic progress tracking (which lessons completed, overall practice time)
- ✅ User authentication and data persistence
- ✅ Responsive web interface

**Explicitly OUT of Scope for MVP:**
- ❌ Real-time conversation (too complex for MVP)
- ❌ Advanced analytics/gamification
- ❌ Social features
- ❌ Mobile apps (web-first)
- ❌ Video/camera features
- ❌ Payment processing (can add post-MVP)

---

## 2. User Experience Flow

### Landing Experience

1. User arrives at homepage
2. Sees clear value prop: "Practice American accent for technical interviews - Get instant AI feedback"
3. Can try one free lesson without signup (reduce friction)
4. After first lesson, prompted to create account to save progress

### Core Learning Flow

```
1. Dashboard/Lesson Library
   ↓
2. Select Scenario (e.g., "Explaining your code in a code review")
   ↓
3. Lesson Page:
   - Context: "You're in a code review. Your teammate asks about your implementation."
   - Key phrases to practice (5-10 phrases):
     * "I implemented this feature using..."
     * "The algorithm complexity is O(n log n)"
     * "I refactored the authentication logic"
   - Example pronunciation tips for tricky words
   ↓
4. Practice Mode:
   - User clicks "Record" for each phrase
   - Speaks into microphone
   - Gets instant feedback:
     * Overall pronunciation score (0-100)
     * Specific words that need work (highlighted)
     * AI tutor tip: "Your 'th' sound in 'authentication' needs work. Try..."
   - Can re-record until satisfied
   ↓
5. Completion:
   - Lesson marked complete
   - Overall score saved
   - Return to dashboard to pick next lesson
```

### Key UX Principles

- **Immediate feedback**: No waiting, instant pronunciation analysis
- **Low pressure**: Practice alone, no embarrassment
- **Clear progress**: See which lessons completed, scores improving
- **Bite-sized**: Each lesson takes 10-15 minutes

---

## 3. Technical Architecture

### Tech Stack

**Frontend:**
- **Framework**: React + TypeScript (leverages founder's frontend expertise)
- **Styling**: Tailwind CSS (rapid development)
- **Audio Recording**: Web Audio API / MediaRecorder API (built-in browser)
- **State Management**: React Context or Zustand (keep it simple for MVP)
- **Hosting**: Vercel (free tier, perfect for Next.js)

**Backend:**
- **Framework**: Next.js API routes (keeps everything in one repo, easier for solo dev)
- **Database**: PostgreSQL on Supabase (free tier, handles auth + DB)
- **File Storage**: Supabase Storage (for audio recordings if needed)

**AI/ML Services:**
- **Speech-to-Text**: OpenAI Whisper API (excellent accuracy, $0.006/minute)
- **Pronunciation Analysis**:
  - Option 1: Compare Whisper transcription with expected text (simple, good enough for MVP)
  - Option 2: Use phonetic libraries like `pronouncing` or `epitran` for deeper analysis
- **AI Tutor Feedback**: Claude API (Sonnet 3.5 - good balance of cost/quality)
- **Text-to-Speech** (for example pronunciations): OpenAI TTS API or ElevenLabs

### Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (React)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Lesson     │  │  Recording   │  │ Progress  │ │
│  │   Library    │  │  Interface   │  │ Dashboard │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              Next.js API Routes                      │
│  /api/lessons          - Get lesson content          │
│  /api/analyze-speech   - Process audio recording     │
│  /api/feedback         - Get AI tutor tips           │
│  /api/progress         - Save/retrieve user progress │
└─────────────────────────────────────────────────────┘
                          ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Supabase   │  │ OpenAI APIs  │  │  Claude API  │
│  (Auth + DB) │  │ (Whisper+TTS)│  │ (AI Tutor)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Data Models

```typescript
// User
interface User {
  id: string
  email: string
  name: string
  created_at: timestamp
  subscription_tier: 'free' | 'pro'
}

// Lesson (stored as JSON/DB)
interface Lesson {
  id: string
  title: string
  category: 'interview' | 'code-review' | 'meeting' | 'presentation'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  context: string // Scenario description
  phrases: Phrase[]
}

interface Phrase {
  id: string
  text: string
  pronunciation_tips: string
  audio_example_url?: string
}

// UserProgress
interface UserProgress {
  id: string
  user_id: string
  lesson_id: string
  phrase_id: string
  recording_url?: string
  score: number // 0-100
  feedback: string
  completed_at: timestamp
}
```

### Key Technical Decisions

**1. Why Next.js full-stack?**
- Single codebase (easier for solo dev)
- API routes handle backend logic
- Great deployment story (Vercel)
- Founder can leverage Claude Code for backend parts they're less familiar with

**2. Why Supabase over custom backend?**
- Auth built-in (email/password, OAuth)
- PostgreSQL with good free tier
- Real-time subscriptions if needed later
- Less infrastructure to manage

**3. Why Whisper for speech recognition?**
- Best-in-class accuracy
- Handles accents well
- Affordable ($0.006/min)
- Simple API

**4. Pronunciation Scoring Approach for MVP:**
- Record user audio → Whisper transcription
- Compare transcription to expected text (word-level accuracy)
- Use phonetic distance for mispronounced words
- Claude generates personalized feedback based on errors
- Good enough for MVP, can enhance later with specialized models

---

## 4. MVP Implementation Plan

### Build Phases

**Phase 1: Core Infrastructure (Week 1-2)**
- [ ] Set up Next.js project with TypeScript
- [ ] Configure Supabase (auth + database)
- [ ] Create basic data models (User, Lesson, UserProgress)
- [ ] Implement authentication (email/password)
- [ ] Deploy to Vercel (get it live early)

**Phase 2: Lesson Content & Display (Week 2-3)**
- [ ] Create 10 initial dev scenarios as JSON:
  1. "Technical Interview: Explaining your approach"
  2. "Code Review: Defending your implementation"
  3. "Daily Standup: Status updates"
  4. "Pair Programming: Discussing solutions"
  5. "Bug Triage: Explaining the issue"
  6. "Architecture Discussion: Proposing a design"
  7. "Onboarding: Introducing yourself to the team"
  8. "Demo Day: Presenting your feature"
  9. "Retrospective: Giving feedback"
  10. "1-on-1: Career discussion with manager"
- [ ] Build lesson library UI (list view)
- [ ] Build lesson detail page (show phrases, context)
- [ ] No recording yet - just display content

**Phase 3: Recording & Basic Feedback (Week 3-4)**
- [ ] Implement audio recording (Web Audio API)
- [ ] Integrate OpenAI Whisper API
- [ ] Build basic pronunciation scoring:
  - Compare Whisper output to expected text
  - Calculate word-level accuracy
  - Highlight mispronounced words
- [ ] Display score + basic feedback

**Phase 4: AI Tutor Enhancement (Week 4-5)**
- [ ] Integrate Claude API for personalized feedback
- [ ] Generate specific pronunciation tips based on errors
- [ ] Add example audio (OpenAI TTS) for correct pronunciation
- [ ] Polish feedback UI

**Phase 5: Progress Tracking (Week 5-6)**
- [ ] Save user recordings and scores
- [ ] Build progress dashboard
- [ ] Show completed lessons, average scores
- [ ] Add "practice again" functionality

**Phase 6: Polish & Launch Prep (Week 6-7)**
- [ ] Responsive design (mobile-friendly)
- [ ] Loading states, error handling
- [ ] Onboarding flow (first-time user experience)
- [ ] Landing page with demo
- [ ] Pricing page (even if not charging yet)

### Development Strategy

**What to Build with Claude Code:**
- ✅ Backend API routes (less familiar territory)
- ✅ Database schema and migrations
- ✅ Supabase integration
- ✅ OpenAI/Claude API integration
- ✅ Audio processing logic

**What to Build Yourself:**
- ✅ Frontend UI/UX (founder's strength)
- ✅ Component design
- ✅ User flows
- ✅ Styling

### MVP Launch Criteria

- [ ] 10 working lessons
- [ ] Recording + pronunciation feedback works
- [ ] Users can create accounts and save progress
- [ ] Deployed and accessible via URL
- [ ] Basic landing page explains value prop

---

## 5. Market Research Insights

### Key Findings from Reddit/X Research

**Successful Solo Founder Patterns:**
- People building SaaS products 99.9% with Claude Code
- Solo developers making $5K-10K/month with AI chatbots and automation
- Non-technical founders building complex software using Claude
- Solopreneurs using Claude + MCPs as their "Head of Marketing"
- One-person teams automating entire business operations with AI agents

**AI Tool Preferences:**
- Claude excels at complex reasoning and long documents
- Developers prefer Claude Code for backend work
- OpenAI Whisper is the standard for speech-to-text
- Gemini gaining traction but Claude still preferred for development

**Market Validation:**
- Conversational AI products generating solid revenue for solo founders
- Language learning is a massive market globally
- Developers are heavy users of AI tools and willing to pay for specialized solutions
- Accent reduction is expensive ($50-100/hour with human tutors) - clear opportunity

---

## 6. Business Model (Post-MVP)

### Pricing Strategy (Future)

**Free Tier:**
- 3 lessons available
- Basic pronunciation feedback
- Limited practice attempts per day

**Pro Tier ($20-30/month):**
- All lessons unlocked
- Unlimited practice
- Detailed pronunciation analytics
- Progress tracking and history
- Priority support

**Enterprise (Future):**
- Team accounts for companies
- Custom lessons for specific industries
- Admin dashboard
- Usage analytics

### Growth Strategy

**Phase 1: Launch to Developer Communities**
- Post on Reddit (r/cscareerquestions, r/learnprogramming, r/ExperiencedDevs)
- Share on Twitter/X with dev hashtags
- Post in Discord communities (frontend, backend, general dev)
- Leverage founder's network for initial users

**Phase 2: Content Marketing**
- Blog posts about technical communication
- YouTube videos demonstrating the platform
- Case studies from early users

**Phase 3: Partnerships**
- Bootcamps and coding schools
- Tech companies with international teams
- Developer communities and conferences

---

## 7. Success Metrics

### MVP Success Criteria

**User Engagement:**
- 100 signups in first month
- 30% of users complete at least 3 lessons
- Average session time > 10 minutes

**Technical Performance:**
- Pronunciation feedback latency < 3 seconds
- 95% uptime
- No critical bugs blocking core functionality

**Qualitative Feedback:**
- 5+ users provide detailed feedback
- At least 3 users express willingness to pay

### Post-MVP Metrics

- Monthly Active Users (MAU)
- Lesson completion rate
- User retention (Day 7, Day 30)
- Average pronunciation score improvement
- Conversion rate (free to paid)
- Monthly Recurring Revenue (MRR)

---

## 8. Risks & Mitigation

### Technical Risks

**Risk**: Speech recognition accuracy for non-native accents
**Mitigation**: Whisper is specifically trained on diverse accents; test with target users early

**Risk**: Pronunciation scoring algorithm not accurate enough
**Mitigation**: Start simple (word-level accuracy), iterate based on user feedback

**Risk**: Audio recording doesn't work on all browsers
**Mitigation**: Use Web Audio API with fallbacks; test on major browsers early

### Business Risks

**Risk**: Users just use ChatGPT/Claude directly instead
**Mitigation**: Specialized scenarios + structured practice + progress tracking = clear differentiation

**Risk**: Market too small (international developers who care about accent)
**Mitigation**: Start narrow, expand to other audiences (business professionals, students) if needed

**Risk**: Solo founder burnout
**Mitigation**: Strict MVP scope, leverage Claude Code for unfamiliar areas, focus on shipping fast

---

## 9. Next Steps

1. **Validate Design** - Review this document, make any final adjustments
2. **Set Up Development Environment** - Initialize Next.js project, Supabase, API keys
3. **Start Phase 1** - Core infrastructure (auth, database, deployment)
4. **Weekly Progress Reviews** - Track against implementation plan
5. **Early User Testing** - Get feedback from 3-5 target users by Week 4
6. **Iterate Based on Feedback** - Adjust features/UX before full launch

---

## 10. Resources & References

### AI Tools for Development
- Claude Code: Backend development, API integration
- OpenAI Whisper: Speech-to-text
- Claude API: AI tutor feedback generation
- OpenAI TTS: Example pronunciation audio

### Technical Documentation
- Next.js: https://nextjs.org/docs
- Supabase: https://supabase.com/docs
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- Whisper API: https://platform.openai.com/docs/guides/speech-to-text

### Market Research Sources
- Reddit communities: r/Entrepreneur, r/Solopreneur, r/ClaudeAI
- X/Twitter: Developer and AI tool discussions
- Competitor analysis: Existing language learning platforms

---

**Document Version**: 1.0
**Last Updated**: 2026-02-10
**Owner**: Frontend Engineer transitioning to Full-stack
**Status**: Ready for Implementation
