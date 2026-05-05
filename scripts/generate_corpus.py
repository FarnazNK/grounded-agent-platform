"""Generate an extended HR corpus for realistic RAG benchmarking.

The default 3-doc corpus is fine for a smoke test but doesn't exercise the
hybrid retrieval pipeline meaningfully — every query matches everything.
This script produces ~60 markdown docs covering HR, IT, security, finance,
benefits, and engineering policies, with deliberate keyword overlap so BM25
and dense retrieval disagree in interesting ways.

Run:
    python scripts/generate_corpus.py
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_corpus_extended"


# ---------------------------------------------------------------------------
# Corpus content. Each entry: (filename, markdown body).
# Written to feel like real internal docs — short, dated, opinionated.
# ---------------------------------------------------------------------------

DOCS: list[tuple[str, str]] = [
    # ----- PTO / Time off -----
    (
        "pto_policy.md",
        """# Paid Time Off (PTO) Policy

Effective: January 1, 2025.

## Accrual
- New hires accrue 15 PTO days per year for the first 2 years.
- Employees with 2-5 years of tenure accrue 20 days per year.
- Employees with 5+ years of tenure accrue 25 days per year.
- PTO accrues monthly, prorated for partial months.

## Carry-over
- Up to 5 days may be carried into the next calendar year.
- Anything above 5 days is forfeited on January 1.
- Carry-over days expire on March 31 of the following year.

## Requesting PTO
- Submit requests in Workday at least 5 business days in advance.
- Manager approval required for any request over 3 consecutive days.
- Blackout periods: last 2 weeks of Q4, first week of January.
""",
    ),
    (
        "sick_leave_policy.md",
        """# Sick Leave Policy

Effective: January 1, 2025.

## Allocation
- All full-time employees receive 10 paid sick days per calendar year.
- Sick days do not accrue; they reset on January 1.
- Unused sick days do not carry over and are not paid out at termination.

## Usage
- Notify your manager via Slack or email before 9 AM the day of absence.
- A doctor's note is required for absences of 3 or more consecutive days.
- Sick leave may be used for the employee or to care for an immediate family member.

## Extended illness
- Beyond 10 days, employees may apply for short-term disability via the HR portal.
- Short-term disability covers up to 12 weeks at 60% of base salary.
""",
    ),
    (
        "parental_leave_policy.md",
        """# Parental Leave Policy

Effective: March 1, 2025.

## Eligibility
- Available to all full-time employees after 6 months of continuous service.
- Applies to birth, adoption, and foster placement.

## Duration
- Primary caregivers: 16 weeks paid leave at 100% of base salary.
- Secondary caregivers: 8 weeks paid leave at 100% of base salary.
- Leave must be taken within 12 months of the qualifying event.

## How to request
- Notify HR at least 30 days in advance when possible.
- Submit form HR-PL-2025 via the HR portal.
- Coordinate handoff plan with your manager 2 weeks before leave begins.
""",
    ),
    (
        "bereavement_leave.md",
        """# Bereavement Leave

- 5 paid days for immediate family (spouse, child, parent, sibling).
- 3 paid days for extended family (grandparent, in-law, aunt, uncle).
- 1 paid day for funeral attendance of close friends or coworkers.
- Additional unpaid leave available with manager approval.
- Notify your manager and HR; no formal documentation required.
""",
    ),
    (
        "jury_duty.md",
        """# Jury Duty Policy

Employees called for jury duty receive paid leave for the full duration of
service. Provide HR with a copy of the jury summons within 5 business days
of receipt. Any compensation received from the court must be reported but
is not deducted from your salary.
""",
    ),
    # ----- Performance & Career -----
    (
        "performance_reviews.md",
        """# Performance Reviews

## Cadence
- Reviews run twice per year: H1 (June) and H2 (December).
- 360-degree feedback collected from peers, reports, and cross-functional partners.

## Ratings
- Exceeds expectations
- Meets expectations
- Partially meets expectations
- Does not meet expectations (triggers PIP — see PIP policy)

## Promotion criteria
To be promoted, an employee must:
1. Receive "Exceeds expectations" in two consecutive cycles, OR receive
   strong calibration support from skip-level and cross-functional partners.
2. Demonstrate sustained impact at the next level for at least one full cycle.
3. Have no active Performance Improvement Plan.
4. Complete the level-up packet (self-assessment, manager assessment, peer
   feedback summary).

Promotion decisions are made at the calibration meeting following each cycle.
""",
    ),
    (
        "pip_policy.md",
        """# Performance Improvement Plan (PIP)

## When a PIP is triggered
A PIP is initiated after a "Does not meet expectations" rating, or at the
manager's discretion with HR approval after sustained underperformance.

## Structure
- Duration: 60 days.
- Weekly 1:1 check-ins with manager.
- Bi-weekly check-ins with HR business partner.
- Specific, measurable goals documented at the start.

## Outcomes
- Successful completion: PIP is closed, employee returns to standard track.
- Unsuccessful completion: separation, with severance per the standard policy.
- Voluntary departure during PIP: severance still applies.
""",
    ),
    (
        "promotion_calibration.md",
        """# Promotion Calibration Process

Calibration meetings happen the week after each performance cycle ends.
Attendees: department head, all managers in the department, HR business
partner. Goal: ensure consistency in promotion decisions across teams.

Each manager presents their proposed promotions with a 5-minute summary.
The room votes by show of hands. Promotions require a 2/3 majority.
""",
    ),
    (
        "compensation_bands.md",
        """# Compensation Bands

We use a level-based compensation model. Each level has a band with a
minimum, midpoint, and maximum. Bands are reviewed annually and adjusted
for market data from Radford and Mercer surveys.

Salary within a band depends on tenure, performance, and location adjustment.
Specific band numbers are available in the HR portal under "Comp Bands".

Mid-cycle salary adjustments require VP approval and are typically reserved
for retention cases or significant scope expansion.
""",
    ),
    (
        "equity_grants.md",
        """# Equity Grant Policy

## Initial grants
New hires receive an initial RSU grant at signing, sized by level.
Grants vest over 4 years with a 1-year cliff: 25% at month 12, then
1/48th monthly thereafter.

## Refresh grants
Top performers receive annual refresh grants in Q1, sized by:
- Performance rating in the previous cycle
- Time since last grant
- Retention risk score (HR-managed)

Refresh grants vest over 4 years with no cliff.
""",
    ),
    # ----- Onboarding -----
    (
        "onboarding.md",
        """# New Hire Onboarding

## Week 1
- Complete compliance training (security, harassment prevention, data handling).
- 1:1 with your manager to align on the 30/60/90 plan.
- Shadow at least 2 cross-functional partners.
- Set up dev environment using the onboarding-dev script.

## Week 2-4
- Ship a small starter task by end of week 2.
- Attend the new-hire welcome lunch (scheduled by the People team).
- Meet with HR to enroll in benefits.

## Day 30
- 30-day check-in with manager.
- Complete the new-hire feedback survey.

## Day 90
- 90-day review with manager and skip-level.
- Confirm probation period closure.
""",
    ),
    (
        "buddy_program.md",
        """# Onboarding Buddy Program

Every new hire is paired with a buddy from a different team. The buddy:
- Meets weekly for the first month, then biweekly through month three.
- Helps the new hire navigate informal norms and tooling.
- Is not in the new hire's reporting chain.

Buddies are nominated by managers and trained in a 1-hour session.
""",
    ),
    (
        "first_day_checklist.md",
        """# First Day Checklist

- [ ] Receive laptop and access badge from IT (front desk)
- [ ] Sign into Okta and set up MFA
- [ ] Join required Slack channels: #announcements, #engineering, your team
- [ ] Update your profile in BambooHR
- [ ] Take welcome photo with the People team
- [ ] Lunch with manager
- [ ] Complete I-9 verification with HR
""",
    ),
    # ----- Remote / Hybrid -----
    (
        "remote_work_policy.md",
        """# Remote Work Policy

## Eligibility
- All full-time employees are eligible for hybrid or remote work.
- Specific arrangements depend on team needs and manager approval.

## Office attendance
- Hybrid employees: 2 days per week in office.
- Fully remote employees: 1 week per quarter on-site at HQ.

## Equipment
- Company provides laptop, monitor, keyboard, mouse, and desk chair.
- One-time $500 home office stipend after 90 days of employment.
- Annual $200 ergonomic refresh stipend.

## International remote
- Working from outside your country of employment requires legal review.
- Maximum 30 days per year of incidental international work without a formal arrangement.
""",
    ),
    (
        "home_office_stipend.md",
        """# Home Office Stipend Details

The home office stipend is $500, payable after 90 days of employment.
Eligible expenses: desk, chair, monitor, monitor arm, lighting, headset.
Not eligible: software subscriptions (use the software stipend instead),
consumables, decorations.

Submit receipts in Concur within 6 months of purchase.
""",
    ),
    (
        "workspace_security.md",
        """# Remote Workspace Security

Working from a non-office location requires:
- Encrypted disk on your work laptop (FileVault or BitLocker).
- VPN connection (AnyConnect) for any access to internal services.
- No company data on personal devices, ever.
- No use of public Wi-Fi without VPN; company-issued mobile hotspot available
  on request.
""",
    ),
    # ----- IT / Security -----
    (
        "password_policy.md",
        """# Password Policy

## Requirements
- Minimum 16 characters.
- Must use 1Password for all work credentials.
- No password reuse across systems.
- Rotation no longer required (per NIST 800-63B); rotate only if compromised.

## MFA
- Required on every system that supports it.
- Hardware tokens (YubiKey) issued to all engineering and security staff.
- SMS-based MFA is deprecated and not allowed for new enrollments.
""",
    ),
    (
        "incident_response.md",
        """# Security Incident Response

## Reporting
- Suspected incident: page the on-call security engineer in PagerDuty
  ("security-oncall" service).
- Phishing: forward the email to phishing@company.example as an attachment.
- Lost device: notify IT immediately and file a ticket in #it-help.

## Severity levels
- SEV1: active breach, customer data exposure. Response: full team paged.
- SEV2: confirmed unauthorized access without exfiltration. Response: team
  paged, public disclosure within 72 hours.
- SEV3: phishing attempt or policy violation. Response: investigated within
  one business day.

## Post-incident
- Blameless postmortem within 5 business days.
- Action items tracked in the security backlog with explicit owners.
""",
    ),
    (
        "data_classification.md",
        """# Data Classification

We use four levels:
- **Public**: marketing site content, published research. No restrictions.
- **Internal**: roadmap docs, design proposals. Default for most docs.
- **Confidential**: customer data, financials, comp data. Need-to-know only.
- **Restricted**: PII, payment data, secrets. Encrypted at rest and in transit;
  access logged and reviewed quarterly.

Each Google Doc and Notion page should have a classification label in the
title or page properties.
""",
    ),
    (
        "acceptable_use.md",
        """# Acceptable Use Policy

Company devices and accounts are for company business. Limited personal use
is acceptable provided it does not interfere with work, consume meaningful
bandwidth, or violate other policies. Prohibited:
- Cryptocurrency mining
- Torrenting
- Storing personal financial or medical records on company systems
- Sharing your account credentials with anyone, including coworkers
""",
    ),
    (
        "software_installation.md",
        """# Software Installation Policy

## Pre-approved software
The Self Service portal lists ~200 pre-approved tools. Install freely.

## Requesting new software
- File a ticket in #it-help with the tool name, vendor, and use case.
- Security review: 3 business days for SaaS, 5 business days for desktop apps.
- AI tools require additional data-handling review.

## Personal accounts
Do not use personal accounts for work-related tools. All subscriptions go
through the procurement process so we can manage data flows centrally.
""",
    ),
    (
        "byod_policy.md",
        """# Bring Your Own Device (BYOD)

BYOD is permitted only for mobile email and Slack via the MDM-enrolled apps.
Personal devices must:
- Be enrolled in MDM (Jamf for iOS, Workspace ONE for Android).
- Have device PIN or biometric unlock enabled.
- Allow remote wipe of company data on departure.

BYOD laptops are not permitted under any circumstance.
""",
    ),
    (
        "vpn_access.md",
        """# VPN Access

Cisco AnyConnect is the company VPN. Required for:
- Access to internal admin tools
- Access to staging and production environments
- Working from public networks

Split tunneling is enabled — only company traffic flows through the VPN.
Personal browsing does not.

Connection issues: see the troubleshooting doc in the IT portal, or page
#it-help.
""",
    ),
    # ----- Engineering -----
    (
        "code_review_policy.md",
        """# Code Review Policy

## Requirements
- All production code requires at least 1 approving review.
- Critical-path code (auth, billing, data export) requires 2 approving reviews.
- Reviews must be from someone other than the author.

## Turnaround
- Reviewers should respond within 1 business day.
- Review-blocked PRs older than 3 days should be raised in the team channel.

## Style
- Mechanical issues (formatting, naming) belong in linters, not review
  comments. If your linter doesn't catch it, file a ticket to fix the linter.
- Prefer concrete suggestions over abstract criticism.
- Disagreements escalate to the tech lead, not to a longer comment thread.
""",
    ),
    (
        "oncall_rotation.md",
        """# On-Call Rotation

## Cadence
- 1 week shifts, Monday 9 AM to the following Monday 9 AM (local time).
- Primary and secondary on-call.
- Engineers join rotation after 90 days of tenure and oncall shadowing.

## Compensation
- $200 per week of primary on-call.
- $50 per page outside business hours.
- Compensatory time off for incidents that span weekends.

## Handoff
- 30-minute handoff meeting every Monday at 9 AM PT.
- Outgoing on-call writes a brief summary in the on-call channel.
""",
    ),
    (
        "incident_postmortems.md",
        """# Incident Postmortems

## When required
- All SEV1 and SEV2 incidents.
- Any SEV3 the team feels was instructive.

## Format
- Blameless. Focus on systems, not individuals.
- Document timeline, root cause, contributing factors, action items.
- Action items must have an owner and a due date.

## Cadence
- Draft within 5 business days of incident resolution.
- Review meeting within 10 business days.
- Public summary published to the engineering postmortem channel.
""",
    ),
    (
        "deploy_freezes.md",
        """# Deployment Freeze Calendar

Production deployments are frozen during:
- Last 2 weeks of December (Dec 18 to Jan 1)
- Day of and day after major product launches
- Last 3 business days before earnings calls

Critical security patches are exempt with VP Engineering approval.
""",
    ),
    (
        "repository_standards.md",
        """# Repository Standards

Every service repository must include:
- README with setup instructions
- CONTRIBUTING.md with the local dev workflow
- CODEOWNERS file
- Working `make test` target
- Working `make lint` target
- Dockerfile (if deployable)
- CI configuration that runs lint, type checks, and tests on every PR

Repos missing these are flagged in the quarterly tech-debt review.
""",
    ),
    (
        "python_style_guide.md",
        """# Python Style Guide (Excerpts)

- Python 3.11+ for new code.
- Ruff for linting and formatting.
- Mypy for type checking.
- Pydantic for data models at boundaries.
- Structlog for logging.
- Tenacity for retry logic.
- Avoid `import *`. Avoid module-level mutable state. Avoid bare `except:`.
- Prefer composition over inheritance for testability.
""",
    ),
    # ----- Finance & Procurement -----
    (
        "expense_policy.md",
        """# Expense Policy

## Limits without pre-approval
- Meals: $50 per person, $100 per team meal.
- Hotels: $300 per night domestic, $400 international.
- Flights: economy domestic, premium economy international over 6 hours.
- Ground transport: rideshare or taxi; rental cars require pre-approval.

## Submission
- Submit receipts in Concur within 30 days.
- Itemized receipts required for any expense over $25.
- Reimbursements processed within 2 weeks of approval.

## Prohibited
- Alcohol on personal expense reports (use the team entertainment budget).
- First-class flights.
- Spa, gym, or wellness purchases (except through the wellness stipend).
""",
    ),
    (
        "travel_booking.md",
        """# Travel Booking

All business travel must be booked through Navan. Booking outside Navan
will not be reimbursed except in documented emergencies.

Rules:
- Domestic flights: book at least 7 days ahead when possible.
- International flights: book at least 21 days ahead when possible.
- Same-day flights require manager approval.

Loyalty program points and miles earned during business travel are yours
to keep.
""",
    ),
    (
        "procurement_process.md",
        """# Procurement Process

Purchases over $1,000 must go through procurement. Steps:
1. File a procurement request in the Procurement portal.
2. Procurement validates vendor (security review, MSA in place, payment terms).
3. Manager and budget owner approve.
4. PO issued; vendor begins work only after PO is in hand.

For SaaS: also include data-handling questionnaire and SOC 2 report.
""",
    ),
    (
        "corporate_card.md",
        """# Corporate Card Policy

- Issued to managers and ICs with regular travel.
- Personal use is prohibited.
- Lost cards: report immediately via the issuer's app and notify Finance.
- Statements reconciled monthly in Concur.
""",
    ),
    # ----- Benefits -----
    (
        "health_insurance.md",
        """# Health Insurance

Plan options through the HR portal during open enrollment (November):
- PPO: highest premium, broad network, no referrals required.
- HDHP with HSA: low premium, $1,500 individual deductible, eligible for
  HSA contributions.
- Kaiser HMO: regional, low premium, requires PCP referrals.

Company contributes 80% of employee-only premium for any plan, 70% for
dependent coverage. Coverage starts on day 1 of employment.
""",
    ),
    (
        "retirement_401k.md",
        """# 401(k) Plan

- Eligibility: day 1 of employment.
- Company match: 100% of the first 4% of salary contributed.
- Vesting: 2-year cliff on the company match (100% vested at month 24).
- Provider: Fidelity. Enroll via the Fidelity portal.
- Roth and traditional contributions both supported.
- Annual contribution limits per IRS guidelines.
""",
    ),
    (
        "wellness_stipend.md",
        """# Wellness Stipend

$1,200 annually, $100 per month, for:
- Gym memberships
- Fitness classes (yoga, climbing, cycling, etc.)
- Mental health apps (Headspace, Calm)
- Therapy not covered by health insurance
- Wearable fitness devices (one per year)

Submit monthly via Forma (replaces Concur for wellness).
""",
    ),
    (
        "learning_stipend.md",
        """# Learning & Development Stipend

$2,000 annually for:
- Books, courses, conferences
- Professional certifications
- Coaching

Pre-approval required for individual expenses over $500. Conferences need
manager approval to confirm time off. Submit via Concur with the L&D
expense category.
""",
    ),
    (
        "commuter_benefits.md",
        """# Commuter Benefits

Pre-tax commuter benefits up to the IRS monthly limit for:
- Public transit passes
- Vanpool
- Eligible parking near the office

Set up via the WageWorks portal during open enrollment or within 30 days
of qualifying life event.
""",
    ),
    (
        "employee_assistance.md",
        """# Employee Assistance Program (EAP)

Free, confidential counseling and referrals through Lyra Health for:
- Mental health (8 free sessions per year per family member)
- Legal consultations (30 minutes free per matter)
- Financial coaching
- Eldercare and childcare resources

Access via the Lyra app. Use of EAP is not visible to your employer.
""",
    ),
    # ----- HR / Employment -----
    (
        "equal_opportunity.md",
        """# Equal Opportunity Statement

We are an equal opportunity employer. Employment decisions are made without
regard to race, color, religion, sex (including pregnancy, sexual orientation,
or gender identity), national origin, age, disability, genetic information,
veteran status, or any other characteristic protected by applicable law.
""",
    ),
    (
        "anti_harassment.md",
        """# Anti-Harassment Policy

Harassment of any kind is not tolerated. Reporting channels:
- Your manager
- Your HR business partner
- The anonymous ethics hotline (number on the back of your badge)
- Skip-level or VP

Investigations are conducted promptly and confidentially. Retaliation against
reporters is itself a terminable offense.
""",
    ),
    (
        "conflict_of_interest.md",
        """# Conflict of Interest

Disclose to HR and your manager any:
- Personal financial interest in a vendor, customer, or competitor.
- Romantic or familial relationship with a coworker in your reporting chain.
- Outside employment that could compete with your work here.

Disclosure does not automatically prohibit the activity; it lets us manage
the conflict appropriately.
""",
    ),
    (
        "outside_employment.md",
        """# Outside Employment

Outside employment is permitted provided:
- It does not compete with company business.
- It does not use company resources, time, or confidential information.
- It is disclosed to your manager.
- It does not create a conflict of interest.

Open source contributions on personal time are explicitly allowed.
Consulting that uses skills developed at the company requires written
approval.
""",
    ),
    (
        "non_compete.md",
        """# Non-Compete Provisions

Per applicable law, we generally do not enforce non-competes for departing
employees. Non-solicitation provisions apply for 12 months after separation:
- No active recruiting of company employees.
- No active solicitation of named customers.

Specific provisions vary by jurisdiction; consult the employment agreement
you signed.
""",
    ),
    (
        "offboarding.md",
        """# Offboarding Process

## Notice period
- Standard: 2 weeks.
- Senior roles (Director+): 4 weeks recommended; negotiable.

## Last day
- Return laptop, badge, and any other company property to IT (or ship via
  the offboarding kit for remote employees).
- HR exit interview (optional but encouraged).
- Final paycheck and unused PTO payout per state law.

## Post-departure
- Email forwarding for 30 days.
- COBRA paperwork mailed within 14 days.
- Stock option exercise window per the equity grant agreement.
""",
    ),
    (
        "references_policy.md",
        """# References Policy

To protect privacy and limit legal exposure, only HR provides references
for former employees. The standard reference confirms:
- Dates of employment
- Title at separation
- Eligibility for rehire (yes/no)

Managers should not provide individual references on behalf of the company.
Personal LinkedIn recommendations are at managers' discretion.
""",
    ),
    # ----- Communication / Meetings -----
    (
        "meeting_norms.md",
        """# Meeting Norms

- Every meeting has an agenda. No agenda, no meeting.
- Default duration: 25 or 50 minutes (build in transition time).
- Default attendance: optional unless explicitly required.
- Recurring meetings should have a stated end date and an explicit purpose.
- Quarterly: every recurring meeting is reviewed for whether it should continue.

Async-first: if a meeting could be a doc, write the doc.
""",
    ),
    (
        "slack_etiquette.md",
        """# Slack Etiquette

- Threads, not new channel messages, for discussion.
- @here only when the channel really needs to see it now.
- @channel only with manager approval; expect to justify it.
- Reactions count as replies. Use them.
- DMs are for things only one person needs to see. Default to channels.
- "You there?" pings are an anti-pattern. Just write your question.
""",
    ),
    (
        "doc_writing_standards.md",
        """# Doc Writing Standards

For decision docs:
1. Context — what's the situation?
2. Decision — what are we doing?
3. Alternatives — what did we consider and reject, and why?
4. Risks — what could go wrong?
5. Owners — who owns the rollout, and who owns reverting?

Keep them short. Two pages max for routine decisions, four pages max for
strategic ones.
""",
    ),
    # ----- Hiring -----
    (
        "hiring_process.md",
        """# Hiring Process

## Steps
1. Recruiter screen (30 min).
2. Hiring manager screen (45 min).
3. Technical screen (1 hour, role-dependent).
4. Onsite loop (4-5 hours): coding, system design, behavioral, manager interview.
5. Debrief and decision (within 2 business days of onsite).

## Candidate experience
- Interviewers commit to 24-hour debrief writeups.
- Decisions communicated to the candidate within 2 business days of the loop.
- Offers extended verbally first, then in writing.
""",
    ),
    (
        "interview_panel_diversity.md",
        """# Interview Panel Composition

Each onsite panel must include:
- At least one woman or non-binary interviewer.
- At least one interviewer outside the immediate hiring team.
- At least one underrepresented-minority interviewer where staffing allows.

When the criteria can't be met, document the exception and the plan to
correct it on future loops.
""",
    ),
    (
        "referral_program.md",
        """# Employee Referral Program

- $5,000 bonus for any referral hired into a non-leadership role.
- $10,000 bonus for engineering and senior+ hires.
- Bonus paid 50% on hire, 50% at the new hire's 6-month mark.
- Self-referrals don't count. Referrals from managers in the candidate's
  reporting chain don't count.
- Submit referrals through the Greenhouse portal.
""",
    ),
    # ----- Misc -----
    (
        "dress_code.md",
        """# Dress Code

There is no formal dress code. Wear what makes you comfortable and is
appropriate for the day's activities. Customer-facing meetings: business
casual. Industry events: brand-appropriate (talk to Marketing if unsure).
""",
    ),
    (
        "pet_policy.md",
        """# Pets in the Office

Dogs welcome at the office on Wednesdays. Requirements:
- Up-to-date vaccinations on file with the office manager.
- Friendly with strangers and other dogs.
- Owner is responsible for cleanup.
- Owner is responsible if the dog disrupts a meeting.

Cats and other animals are not currently permitted in the office.
""",
    ),
    (
        "office_amenities.md",
        """# Office Amenities

Each office offers:
- Free snacks and beverages
- Catered lunch on Tuesdays and Thursdays
- Quiet rooms for focused work
- Phone booths for calls
- Mother's room (book in Outlook)
- Wellness room (lights-off relaxation, no booking required)
- Bike storage

Specific amenity availability varies by location; see the office wiki page.
""",
    ),
    (
        "holiday_calendar.md",
        """# Company Holidays

Observed holidays (US):
- New Year's Day
- Martin Luther King Jr. Day
- Presidents' Day
- Memorial Day
- Juneteenth
- Independence Day
- Labor Day
- Thanksgiving and the day after
- Christmas Eve and Christmas Day

Plus 2 floating holidays per calendar year, usable for cultural or religious
observances of your choice. Floating holidays do not carry over.
""",
    ),
    (
        "internal_transfers.md",
        """# Internal Transfers

Employees in good standing for at least 12 months in their current role may
apply for internal transfers. Process:
1. Notify your current manager of intent.
2. Apply through Greenhouse like an external candidate.
3. Interview with the receiving team.
4. If selected, transition timeline negotiated between managers (typically
   2-4 weeks).

Internal transfers do not reset PTO accrual or vesting schedules.
""",
    ),
    (
        "relocation_policy.md",
        """# Relocation Policy

For employees relocating at the company's request:
- Moving costs covered up to $15,000 (domestic) or $30,000 (international).
- Temporary housing for up to 60 days.
- One house-hunting trip for the employee and a partner.
- Tax gross-up applied per current IRS treatment.

Voluntary relocations (employee request) may receive partial assistance at
manager discretion.
""",
    ),
    (
        "intellectual_property.md",
        """# Intellectual Property Assignment

Per the employment agreement, work created in the scope of your job is
assigned to the company. This does not cover:
- Work created on personal time, on personal equipment, unrelated to company business.
- Open source contributions made under the company's open source policy.
- Pre-existing inventions disclosed at hire and listed in the IP exclusion schedule.

If you're unsure whether something falls inside or outside the scope,
ask the legal team before publishing.
""",
    ),
    (
        "open_source_policy.md",
        """# Open Source Contributions

Encouraged. Process:
1. For personal-time contributions to projects unrelated to your work,
   no approval needed.
2. For contributions during work hours, or to projects we use in production,
   file a request in the OSPO portal.
3. For releasing company-developed code as open source, the OSPO process
   is mandatory and includes a license review and security review.

Contact: ospo@company.example.
""",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in DOCS:
        (OUT_DIR / name).write_text(body, encoding="utf-8")
    print(f"Wrote {len(DOCS)} docs to {OUT_DIR}")


if __name__ == "__main__":
    main()
