# Lumen Retail — E-Commerce Replatform
## Proposal

Prepared for Lumen Retail Group. Submitted January 2024.

## Executive Summary

Lumen Retail's storefront runs on a heavily customised Magento 1 installation that reached end of life in June 2020. The site carries 34,000 SKUs and did GBP 18 million in online revenue last year. Page load times average 4.6 seconds on mobile, and the checkout abandonment rate is 79%.

This proposal covers a replatform to a headless commerce architecture, delivered as a fixed bid over five months.

## Scope of Work

Storefront rebuild as a headless Next.js application with server-side rendering and incremental static regeneration for category and product pages.

Commerce backend migration to Commercetools, including the full catalogue of 34,000 SKUs, customer accounts, and three years of order history.

Checkout rebuild with Stripe as the payment processor, supporting card payments, Apple Pay, Google Pay and Klarna. The current checkout is five steps; the replacement is two.

Integration with Lumen's existing NetSuite ERP for inventory and order fulfilment, and with Klaviyo for marketing automation.

Out of scope: the physical point-of-sale system, the warehouse management system, and any redesign of the brand identity. We work to Lumen's existing design language.

## Technology Stack

The storefront is Next.js 14 with TypeScript, deployed on Vercel. Commercetools provides the commerce engine. Stripe handles payments. PostgreSQL holds the customer preference and content data that does not belong in Commercetools. A small Node.js middleware service brokers the NetSuite integration, since NetSuite's SuiteTalk API cannot be called safely from the edge.

We recommend Commercetools over a monolithic platform because Lumen's roadmap includes a marketplace model in 2025, and the composable architecture makes that an extension rather than a second replatform.

## Delivery Plan

Phase 1, weeks 1 to 4. Catalogue data modelling, Commercetools setup, and the migration tooling for SKUs and customer records. A test migration runs at the end of week 4.

Phase 2, weeks 5 to 12. Storefront build: home, category, search, product detail, cart and account pages. NetSuite and Klaviyo integrations.

Phase 3, weeks 13 to 18. Checkout and payments, order history migration, performance tuning, and SEO parity validation against the existing site.

Phase 4, weeks 19 to 22. Soft launch to 10% of traffic, monitoring, then full cutover. Two weeks of hypercare follow cutover.

Total duration is 5 months.

## Team Structure

Seven people: one engagement lead, one technical architect, three full-stack engineers, one commerce specialist, and one QA engineer.

## Commercials

Fixed bid of USD 180,000. This includes all engineering, QA and project management for the scope described. It excludes Commercetools, Vercel and Stripe licence and transaction fees, which Lumen contracts directly.

Payment milestones: 25% on signature, 25% on Phase 2 acceptance, 30% on Phase 3 acceptance, 20% on successful cutover.

## Risks and Assumptions

SEO regression is the material commercial risk in any replatform. We hold URL structure constant, migrate all existing redirects, and validate Core Web Vitals and structured data against the current site before cutover. We have budgeted a week in Phase 3 specifically for SEO parity work.

We assume Lumen's catalogue data is exportable from Magento in a consistent form. Our experience is that roughly 5% of SKUs in a catalogue this age have data quality issues requiring manual review; we have allowed for that and Lumen's merchandising team resolves the exceptions.

We assume NetSuite API credentials and a sandbox are available by week 2.

## Quality Assurance

Automated test coverage targets the revenue path first: search, product detail, add to cart, and checkout run as end-to-end tests on every deployment. Visual regression testing runs on the top 40 templates. Load testing validates the storefront at three times Lumen's recorded peak traffic before cutover.
