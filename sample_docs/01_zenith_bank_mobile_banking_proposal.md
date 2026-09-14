# Zenith Bank — Mobile Banking Platform Rebuild
## Proposal

Prepared for Zenith Bank Ltd. Submitted March 2023.

## Executive Summary

Zenith Bank's existing mobile application was built in 2016 on a hybrid Cordova stack and now serves 2.1 million active customers. Crash rates have reached 4.8% of sessions and the App Store rating has fallen to 2.9. This proposal covers a complete rebuild of the customer-facing mobile application and the supporting API layer, delivered as a fixed-bid engagement over nine months.

We have delivered eleven regulated fintech platforms since 2018, including three core banking integrations. Our proposed team has shipped React Native applications at comparable scale for institutions in the UK and UAE.

## Scope of Work

The engagement covers the following workstreams.

Customer mobile application for iOS and Android, rebuilt natively in React Native with a shared design system. Screens in scope: onboarding and KYC, account overview, transaction history with search, domestic and international transfers, bill payments, card controls, and in-app support.

API and integration layer built in Node.js, exposing a consolidated GraphQL gateway over Zenith's existing Finacle core banking system, the Salesforce CRM, and the fraud engine. The gateway isolates the mobile clients from core banking downtime through a read-through cache.

Customer onboarding and KYC flow with document capture, liveness detection, and integration with the incumbent identity verification provider. Regulatory requirement: full audit trail on every identity decision, retained for seven years.

Out of scope: the corporate banking portal, ATM software, and any change to the Finacle core itself.

## Technology Stack

The mobile applications are built in React Native 0.71 with TypeScript. The API gateway is Node.js 18 with Apollo Server, deployed on AWS ECS Fargate behind an Application Load Balancer. PostgreSQL 14 on RDS holds session, preference and audit data. Redis provides the read-through cache over core banking calls. Infrastructure is defined in Terraform. CI/CD runs on GitHub Actions with Fastlane for mobile release automation.

We selected React Native over separate native codebases because Zenith's internal team is small and a shared codebase keeps ongoing maintenance within their capacity. Where platform-specific behaviour is unavoidable — biometric authentication, push notification handling, and the card control widgets — we drop to native modules.

## Delivery Plan

Phase 1, Discovery and Foundations, weeks 1 to 6. Design system, architecture spike on the Finacle integration, CI/CD pipeline, and a walking skeleton covering login through account overview.

Phase 2, Core Banking Features, weeks 7 to 22. Accounts, transactions, transfers, bill payments and card controls. Two-week sprints with a demo to Zenith stakeholders at the end of each.

Phase 3, KYC, Hardening and Launch, weeks 23 to 36. Onboarding and KYC flow, penetration testing remediation, performance tuning, phased rollout to 5%, 25% and then 100% of users.

Total duration is 9 months from contract signature.

## Team Structure

The delivery team is eleven people: one engagement manager, one technical architect, four React Native engineers, three Node.js engineers, one QA automation engineer, and one UX designer. The engagement manager and architect are allocated part-time from month four onward.

## Commercials

This is a fixed bid of USD 450,000, inclusive of all roles listed above and all AWS costs during development. Zenith assumes AWS costs from production launch onward.

Payment milestones are structured as follows. 20% on contract signature. 25% on Phase 1 acceptance. 30% on Phase 2 acceptance. 25% on production launch and thirty days of stable operation.

The fixed bid assumes the scope described above. Change requests are estimated separately and charged at USD 85 per hour.

## Risks and Assumptions

The primary risk is the Finacle integration. Zenith's core banking vendor has historically taken six to eight weeks to provision sandbox access. We have assumed sandbox availability by end of week 3; each week of delay beyond that moves the launch date by one week.

We assume Zenith provides a single empowered product owner available for at least ten hours per week, and that the existing identity verification contract permits mobile SDK use.

Penetration testing is conducted by Zenith's nominated third party. We have budgeted two weeks for remediation; findings requiring architectural change are treated as a change request.

## Quality Assurance

Every pull request runs unit tests, static analysis and a dependency vulnerability scan. We target 80% unit coverage on business logic. End-to-end tests run nightly on Detox against a device farm covering the twelve device and OS combinations that represent 91% of Zenith's user base. Release candidates go through a two-day exploratory testing cycle before submission.

## Security and Compliance

The application is built to OWASP MASVS Level 2. Certificate pinning, jailbreak and root detection, and encrypted local storage are included. No customer data is persisted on the device beyond an encrypted session token. All API traffic is TLS 1.3. The audit trail for identity decisions is written to an append-only store in PostgreSQL with row-level integrity hashing.
