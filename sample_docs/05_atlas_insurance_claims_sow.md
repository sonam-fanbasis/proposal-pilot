# Atlas Insurance — Claims Automation Platform
## Statement of Work

Client: Atlas Insurance PLC. Effective date: 15 February 2023.

## 1. Background

Atlas Insurance processes approximately 240,000 motor and property claims per year. First notification of loss arrives through four channels: a call centre, a broker portal, email and post. Claims handlers currently re-key data from all four into a mainframe claims system. Average time from notification to first handler action is 3.2 days.

## 2. Objectives

Reduce time from first notification of loss to first handler action to under four hours for straight-through-processable claims. Eliminate manual re-keying for the broker portal and email channels. Provide handlers with a single work queue across all channels.

## 3. Scope of Services

### 3.1 Intake and Normalisation

A claims intake service that accepts submissions from all four channels and normalises them into a single canonical claim model. Email intake uses document classification and extraction to pull structured data from attached forms and correspondence.

### 3.2 Triage and Routing

A rules-driven triage engine that scores each claim for complexity and fraud indicators and routes it either to straight-through processing or to a handler queue segmented by specialism and workload.

### 3.3 Handler Workspace

A web workspace presenting the handler's queue, the full claim record, document viewer, and the actions available at the current claim stage. The workspace writes back to the existing mainframe claims system through a message-based integration layer.

### 3.4 Exclusions

Replacement of the mainframe claims system of record, changes to underwriting systems, and the customer-facing claims portal are excluded from this Statement of Work.

## 4. Technology

Services are implemented in Java 17 with Spring Boot. Apache Kafka provides the event backbone between intake, triage and workspace services. PostgreSQL holds the canonical claim model and operational data. The handler workspace is built in Angular. Document classification and extraction use a fine-tuned model served from a containerised inference service.

Deployment is to Atlas's existing on-premise Kubernetes platform. Atlas's regulatory position does not permit claims data to leave their own data centres, which rules out public cloud for this workload.

## 5. Engagement Model and Commercials

Services are provided on a dedicated team basis. The team comprises one delivery manager, one solution architect, six backend engineers, two front-end engineers, one machine learning engineer, and two QA engineers.

The engagement runs for 18 months from the effective date. Total contract value is USD 920,000, invoiced monthly in arrears at USD 51,111 per month.

Either party may terminate with 60 days written notice. On termination Atlas receives all source code, infrastructure definitions and documentation produced to that date.

## 6. Delivery Milestones

Milestone 1, month 3: intake service accepting broker portal and email channels into the canonical model, in a non-production environment.

Milestone 2, month 7: triage engine operational with the initial rule set, and mainframe write-back integration proven end to end.

Milestone 3, month 11: handler workspace in production for the motor claims line.

Milestone 4, month 15: property claims line live, all four intake channels in production.

Milestone 5, month 18: straight-through processing enabled, handover and knowledge transfer complete.

## 7. Assumptions

Atlas provides mainframe integration specifications and a test environment by week 4. Atlas provides a representative corpus of at least 20,000 historical claim documents for model training by week 6. Atlas provides a product owner and a claims operations subject matter expert, each available a minimum of two days per week.

## 8. Service Levels During Hypercare

Following each production milestone, a 30-day hypercare period applies. Severity 1 incidents, meaning a total loss of claims intake, carry a one-hour response target and a four-hour resolution target. Severity 2 incidents carry a four-hour response and one-business-day resolution target.
