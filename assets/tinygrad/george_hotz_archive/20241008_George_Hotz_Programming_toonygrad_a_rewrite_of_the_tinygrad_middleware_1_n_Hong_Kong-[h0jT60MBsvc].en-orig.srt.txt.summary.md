# George Hotz Programming Stream Summary

This is a transcript from George Hotz (geohot) livestreaming his work on "toonygrad", a rewrite of the middleware components from his project "tinygrad".

## Personal and Political Commentary
- George has moved to Hong Kong and discusses his disillusionment with American politics
- Shares thoughts on differences between US and China regarding business environment, immigration, and infrastructure
- Expresses concerns about US policies, particularly regarding immigration and taxation

## Technical Content: toonygrad Development
- George is rewriting the middleware of tinygrad to create "toonygrad"
- The project revolves around a system of "UOPs" (universal operations) as the core abstraction
- Main implementation work includes:
  - Building a graph-based computational framework
  - Creating tensor operations that compile to efficient code
  - Implementing pattern matching and rule-based graph optimizations
  - Writing a visualization system to display computational graphs

## Core Technical Concepts Demonstrated:
1. Graph-based computation with operations like Swizzle, ALU, reshape, etc.
2. Pattern matching system that rewrites and optimizes operations
3. Automatic constant folding and operation simplification
4. Breaking computations into kernels for efficient execution

## Development Philosophy
- Emphasizes building software with a 10-20 year lifespan rather than quick releases
- Focuses on getting abstractions right before worrying about adoption
- Compares the clean design of tinygrad/toonygrad with more complex frameworks

The stream concludes with George mentioning an upcoming tinygrad team meeting and discussing his thoughts on Hong Kong vs. the US as places to live and work.