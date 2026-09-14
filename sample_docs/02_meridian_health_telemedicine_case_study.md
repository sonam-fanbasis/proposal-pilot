# Meridian Health — Telemedicine Platform
## Case Study

Delivered 2024. Client: Meridian Health Group.

## The Problem

Meridian Health operates 34 outpatient clinics across three states. During 2022 their video consultation volume grew from 400 to 11,000 appointments per month, running on an off-the-shelf video product that was never designed for clinical workflow. Clinicians were copying consultation notes by hand into the electronic health record after every call. The average consultation carried 7 minutes of administrative overhead, and the platform had no path to HIPAA compliance attestation.

## What We Built

We built a purpose-built telemedicine platform integrated directly with Meridian's Epic electronic health record system.

The clinician workspace presents the patient chart, the video call and the note editor in a single screen. Consultation notes write back to Epic through FHIR APIs on call completion, eliminating the manual re-entry step entirely.

The patient application handles appointment booking, pre-consultation intake questionnaires, waiting room, the consultation itself, and post-visit summary and prescription delivery.

An asynchronous messaging channel lets patients send follow-up questions to their care team within a defined clinical window, with automatic escalation when a message goes unanswered for four hours.

## Technology Stack

The backend is Python with Django and Django REST Framework, on PostgreSQL 15. Real-time video uses WebRTC with a self-hosted SFU rather than a third-party video vendor, which was necessary because Meridian's compliance team required that no media ever traverse infrastructure outside the signed BAA. The front end is React with TypeScript. Everything runs on Microsoft Azure in a HIPAA-eligible configuration, orchestrated with Kubernetes on AKS. Asynchronous work runs on Celery with Redis. Integration with Epic uses FHIR R4.

## Engagement Model

This was a dedicated team engagement rather than a fixed bid. Meridian's requirements were genuinely uncertain at the start — the clinical workflow had never been mapped — and a fixed bid would have forced premature specification. The dedicated team model let us run clinical discovery in parallel with delivery.

The team ran for 14 months: one delivery lead, six engineers, one QA engineer, one designer and a part-time clinical analyst. Total engagement value was USD 680,000.

## Delivery Timeline

Clinical discovery and workflow mapping ran through months 1 and 2, shadowing 40 consultations across six specialties. Platform foundations and the Epic FHIR integration spike covered months 3 to 5. The clinician workspace was built between months 6 and 9, and the patient application between months 8 and 11. HIPAA readiness, penetration testing and the SOC 2 evidence pack ran through months 12 to 14, with a pilot at four clinics from month 12.

## Security and Compliance

The platform achieved HIPAA compliance attestation in month 14. All protected health information is encrypted at rest with customer-managed keys in Azure Key Vault and in transit with TLS 1.3. Access is role-based with break-glass emergency access that raises an immediate alert and a mandatory post-hoc review.

Every read of a patient record is written to an immutable audit log. The log is queryable by Meridian's compliance officers and retained for seven years.

We completed a third-party penetration test in month 13. Four medium findings and no high or critical findings were identified; all four were remediated within the test window.

## Results

Administrative overhead per consultation fell from 7 minutes to 40 seconds, measured across 4,200 consultations in the three months after launch. That represents roughly 1,100 clinician hours returned per month at current volume.

Patient no-show rate fell from 18% to 11%, which Meridian attributes to the automated reminder and one-tap join flow.

The platform now handles 14,000 consultations per month. Median call setup time is 3.1 seconds and call completion rate is 98.7%.

Meridian retired the previous video vendor contract, saving USD 96,000 annually.
