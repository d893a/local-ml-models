# George Hotz AMD 7900XTX Research Stream Summary

This transcript captures George Hotz (geohot) exploring and documenting the AMD 7900XTX GPU during a livestream. Key points:

## AMD Open Source Claims
- George is skeptical about AMD's promises to open source their GPU drivers
- Criticizes AMD for making announcements without releasing actual code
- "A tweet and a blog post are not anything... I want to see source code"
- Questions whether AMD truly understands "the spirit of Open Source"

## Technical Investigation
- Investigating GPU crashes that may be firmware-related
- Exploring PM4 packets as an alternative to AQL for kernel submission
- Uses UMR (User Mode Register debugger) to inspect GPU state
- Discovers a possible firmware regression - older versions seem more stable
- Spends time trying to understand memory queue descriptors (MQDs) and pipe queues (PQs)

## Philosophical Commentary
- References "hacker culture" values and The Hacker Manifesto
- "No problem should ever have to be solved twice"
- Criticizes intellectual property laws, especially copyright extension acts
- Argues for quality engineering over marketing/PR

## Tools and Environment
- Struggles with X11 forwarding and VNC to access GUI tools
- Eventually gets UMR GUI working through Asahi Linux on a MacBook
- Praises Asahi Linux's code quality and engineering approach

## Personal Responses
- Addresses criticism about his communication style and ego
- Defends his technical approach and contributions
- Emphasizes making things that work rather than focusing on PR

The stream demonstrates his hands-on approach to understanding hardware at a low level while providing commentary on open source culture, engineering ethics, and his technical philosophy.