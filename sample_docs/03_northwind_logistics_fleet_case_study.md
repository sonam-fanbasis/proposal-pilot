# Northwind Logistics — Real-Time Fleet Tracking
## Case Study

Delivered 2022. Client: Northwind Logistics BV.

## The Problem

Northwind operates 2,400 vehicles across Northern Europe. Their dispatch team was working from position data that averaged 11 minutes old, sourced from a telematics vendor's polling API. Customers calling about a delivery received estimates that were routinely wrong by an hour or more, and the dispatch team had no way to reroute around incidents in time to matter.

## What We Built

A real-time fleet tracking and dispatch platform that ingests telematics events directly rather than polling for them.

The ingestion pipeline accepts position, speed, fuel and engine-diagnostic events from three different telematics hardware vendors, normalises them into a single event schema, and makes them queryable within two seconds of the vehicle emitting them.

The dispatch console shows live vehicle positions on a map with predicted arrival times recalculated continuously from live traffic data. Dispatchers can reassign a stop to a different vehicle and the affected drivers receive the updated manifest immediately.

A customer-facing tracking page gives end recipients a live map and an arrival window, which removed a large share of inbound "where is my delivery" calls.

## Technology Stack

The ingestion and routing services are written in Go, chosen for throughput per instance — the platform sustains 18,000 events per second at peak on six instances. Apache Kafka is the event backbone, with separate topics per event class and a 72-hour retention window. PostgreSQL with PostGIS holds geospatial and operational data. Redis caches the live vehicle state that the dispatch console polls. The dispatch console front end is React.

Everything runs on Google Cloud Platform, on Kubernetes via GKE, with Terraform-managed infrastructure. Observability is Prometheus and Grafana with distributed tracing in Jaeger.

## Engagement Model

Northwind engaged us on a time and materials basis. The integration surface across three telematics vendors was unknown at the outset — two of the three vendors had no usable documentation — and neither side was willing to price that uncertainty into a fixed bid.

The engagement ran 7 months and totalled USD 320,000 at a blended rate of USD 78 per hour. Northwind received a weekly burn report and could stop or redirect the work at any sprint boundary.

## Team Structure

Five people throughout: one technical lead, three backend engineers with Go and distributed systems experience, and one front-end engineer. Northwind provided their own QA resource and a dedicated product owner from the dispatch operations team.

## Delivery Timeline

Vendor integration discovery and the event schema design took weeks 1 to 5. The Kafka ingestion pipeline and normalisation layer ran weeks 4 to 12. The dispatch console was built weeks 10 to 22. The customer tracking page and public API covered weeks 20 to 26. A pilot with 120 vehicles ran from week 18, expanding to the full fleet by week 28.

## Risks We Managed

The largest risk materialised as expected: the second telematics vendor's event stream turned out to be undocumented and inconsistent, emitting position events with timestamps in three different timezone conventions depending on hardware revision. We built a normalisation layer with per-device-revision handling and a quarantine queue for events that fail validation. Roughly 0.3% of events land in quarantine and are reviewed weekly.

## Results

Position data latency fell from an 11-minute average to under 2 seconds at the 95th percentile.

Arrival-time accuracy within a 15-minute window improved from 54% to 89%, measured over 140,000 deliveries in the quarter following full rollout.

Inbound customer service calls about delivery status dropped 62%, which Northwind valued at approximately EUR 340,000 per year in call-centre cost.

Fuel consumption fell 4.1% in the first six months, attributed to the rerouting capability and idle-time alerting.
