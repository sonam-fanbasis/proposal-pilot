# Engineering Capability Statement

Internal reference document. Last revised 2024.

## About Our Delivery Organisation

We are a software engineering partner of 340 people, of whom 280 are billable engineers. We have delivered 190 client engagements since 2015 across financial services, healthcare, logistics, retail and insurance. Our average engagement length is 11 months and our client retention rate past the first engagement is 74%.

Delivery is organised into cross-functional pods rather than resource pools. A pod stays with an engagement from discovery to handover, which is why our engagements report a mean of 1.4 team changes per year against an industry norm closer to 4.

## Engagement Models

We work under three commercial models and recommend the one that fits the uncertainty in the work rather than the one that is easiest to sell.

Fixed bid suits engagements where scope can be specified with confidence before work starts — typically replatforms, migrations, and well-understood rebuilds. We carry the estimation risk. We will not quote fixed bid on work where requirements are genuinely unknown, because the only way to protect a fixed price against unknown scope is to fight every change request, and that is a bad outcome for both sides.

Time and materials suits exploratory work, integration against undocumented systems, and R&D. Clients receive a weekly burn report and can redirect or stop work at any sprint boundary.

Dedicated team suits long-running product development where the client wants continuity and the roadmap will evolve. The team is ring-fenced to the client, and the client's product owner sets the backlog directly.

## Quality Assurance Approach

Testing is part of the definition of done, not a phase that follows development. Every pull request must pass unit tests, static analysis, and a dependency vulnerability scan before merge. We target 80% unit test coverage on business logic and do not chase coverage on generated or presentational code, because coverage targets applied indiscriminately produce tests that assert nothing.

Each engagement defines a revenue-path or critical-path test suite — the handful of end-to-end journeys where a regression is unacceptable — and those run on every deployment. Broader regression suites run nightly.

We staff a dedicated QA automation engineer on every engagement above four people. On regulated work we add exploratory testing cycles before each release candidate, because automated suites test what we thought to specify and exploratory testing finds what we did not.

Performance testing is scoped to the engagement's actual traffic profile and run against a production-like environment before any major release. We validate at three times recorded peak as a default.

## Security Practice

All engineers complete secure development training annually. We build to OWASP Application Security Verification Standard Level 2 by default, and Level 3 where the client's regulatory position requires it.

Every engagement includes threat modelling during the architecture phase and a security review before each production release. We integrate SAST and dependency scanning into CI and treat a high-severity finding as a build failure, not a ticket.

We support third-party penetration testing and budget remediation time explicitly rather than assuming a clean result. Across engagements in the last three years, third-party tests have averaged 3.1 medium findings and 0.2 high findings per engagement, with median remediation time of six working days.

We hold ISO 27001 certification and complete SOC 2 Type II evidence packs where a client requires one.

## Compliance Experience

We have delivered under HIPAA for healthcare clients in the United States, GDPR across our European engagements, PCI DSS for payment-handling systems, and FCA regulatory requirements for UK financial services clients. We have supported two clients through SOC 2 Type II attestation.

## Cloud and Infrastructure

We are certified partners on AWS and Google Cloud and have delivered production workloads on Microsoft Azure. We also deliver to on-premise Kubernetes platforms, which matters for clients whose regulatory position does not permit public cloud.

Infrastructure is defined as code without exception, using Terraform. We do not hand over environments that were configured by hand, because they cannot be rebuilt and they cannot be audited.

## Knowledge Transfer and Handover

Every engagement includes handover as a funded work item rather than a goodwill gesture at the end. Handover comprises architecture documentation, runbooks for each operational scenario, a recorded walkthrough of each subsystem, and a period of paired working with the client's own engineers.

We do not consider an engagement complete until the client's team has independently deployed the system to production at least once.
