# Routing benchmark on the real stack: 20261114-214821

- made by bench/fault_table.py against http://router:8000 (model afno-19m)
- trace: seed 42, 1182 requests over 150 s (about 8.0 requests a second), the same for every rule
- events, the same for every rule: 10:slow:box-a,90:normal:box-a,105:kill:box-b
- slow = a real fine-tune on that box (the supervisor's job, with --out /wx/ft/fault-table); kill = the forecast server's process is killed and its supervisor starts it again
- latency = what the client saw, answered requests only; errors = no answer or not HTTP 200

| policy | reqs | errors | err % | p50 ms | p95 ms | p99 ms | max ms | mean ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 1182 | 0 | 0.0 | 3074.7 | 26892.8 | 28612.7 | 29738.9 | 8333.1 | 100.0 | 0.0 | 0 |  |
| round_robin | 1182 | 0 | 0.0 | 73.9 | 509.0 | 1220.2 | 1795.3 | 142.8 | 54.1 | 45.9 | 1 |  |
| least_loaded | 1182 | 0 | 0.0 | 80.7 | 387.4 | 873.1 | 1170.7 | 125.2 | 49.4 | 50.6 | 0 |  |
| adaptive | 1182 | 0 | 0.0 | 108.8 | 1004.2 | 2309.6 | 2792.4 | 264.1 | 45.6 | 54.4 | 6 | 6 |
| balanced | 1182 | 0 | 0.0 | 81.9 | 410.7 | 792.7 | 1605.6 | 130.8 | 46.4 | 53.6 | 0 |  |

p95 by window of the case (requests in brackets). "training" starts at the job's first training step; "box down" ends when the killed server answers again.

| policy | before p95 ms | job starting p95 ms | training p95 ms | between p95 ms | box down p95 ms | after p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| static | 1067.3 (89) | 3010.5 (279) | 17915.9 (368) | 29550.8 (82) | 27213.9 (110) | 19551.0 (254) |
| round_robin | 414.5 (89) | 277.3 (219) | 873.4 (442) | 904.3 (68) | 211.9 (90) | 175.1 (274) |
| least_loaded | 282.2 (89) | 280.9 (211) | 313.1 (436) | 288.2 (82) | 879.4 (95) | 259.6 (269) |
| adaptive | 1068.6 (89) | 591.3 (199) | 1223.9 (449) | 1212.7 (81) | 204.2 (114) | 425.1 (250) |
| balanced | 609.8 (89) | 295.2 (215) | 411.8 (432) | 251.3 (82) | 343.3 (90) | 427.4 (274) |

**static**: first training step at 45.03 s, job stopped at 93.56 s, killed box down at 105.06 s and answering again at 118.07 s (the router used it again at 118.57 s); the sender was at most 2.2 ms late; 0 unanswered

**round_robin**: first training step at 38.02 s, job stopped at 95.55 s, killed box down at 105.06 s and answering again at 116.07 s (the router used it again at 116.57 s); the sender was at most 2.8 ms late; 0 unanswered

**least_loaded**: first training step at 36.53 s, job stopped at 93.56 s, killed box down at 105.07 s and answering again at 116.58 s (the router used it again at 117.08 s); the sender was at most 9.0 ms late; 0 unanswered

**adaptive**: first training step at 34.53 s, job stopped at 93.58 s, killed box down at 105.09 s and answering again at 118.59 s (the router used it again at 119.09 s); the sender was at most 2.2 ms late; 0 unanswered

adaptive switches:
- 616.62 s box-a -> box-b (box-b 18ms vs box-a 32ms)
- 634.67 s box-b -> box-a (box-a 18ms vs box-b 56ms)
- 645.69 s box-a -> box-b (box-b 18ms vs box-a 33ms)
- 678.77 s box-b -> box-a (box-a 40ms vs box-b 2004ms)
- 688.8 s box-a -> box-b (box-b 18ms vs box-a 39ms)
- 728.9 s box-b -> box-a (active box down)

**balanced**: first training step at 37.53 s, job stopped at 93.57 s, killed box down at 105.08 s and answering again at 116.09 s (the router used it again at 116.59 s); the sender was at most 3.9 ms late; 0 unanswered
