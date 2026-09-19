# Executive Summary

**Headline Finding**  
An analysis of 55,000 execution runs identified an overall failure rate of 15.0%, primarily driven by host temperature (12.8% feature importance), seed group (8.1%), feature flag 7 (3.6%), voltage (3.6%), and memory allocation (3.3%). Out of 24 tested system configurations, 22 are strictly dominated across both performance and reliability, leaving only 2 viable, Pareto-optimal configurations for leadership to consider.

**Root Cause Story**  
Failures are driven by specific configuration flags, seed groups, and resource constraints rather than broad test noise. Enabling `feature_flag_7` is the largest single contributor, accounting for 49.1% of all failures (4,058 total). Furthermore, while 50 of 51 tested randomization variables (including workload type and traffic pattern) were ruled out as noise, `seed_group` 8 is statistically significant (p=0.0000) and triggers a median 8x+ failure rate spike, generating 28.6% of total failures (2,365 total). Low memory allocation combined with high thread pool sizes (memory_alloc_mb <= 1024 AND thread_pool_size >= 16) accounts for another 29.7% of failures (2,459 total). Log signals confirm these drivers, showing failed runs suffer from high increases in scheduler errors (10.04x fail-to-pass ratio), general errors (9.08x ratio), and memory errors (8.04x ratio).

**Actionable Recommendation**  
Leadership must evaluate a direct trade-off between throughput and stability, as both top-performing options utilize an Adaptive cache and Dynamic scheduler but differ on `feature_flag_7`:
*   **Lowest Failure Risk:** Disabling the flag (`feature_flag_7 = False`) yields a 9.19% failure rate and 1583.1 throughput (based on 3,776 historical runs).
*   **Highest Throughput:** Enabling the flag (`feature_flag_7 = True`) increases throughput to 1765.8, but accepts a higher failure rate of 19.77% (based on 2,069 historical runs).

**Confidence & Limitations**  
While the overall weighted repeatability score of 0.896 indicates high general determinism (1.0 scale), this average masks severe localized variance caused specifically by seed group 8.