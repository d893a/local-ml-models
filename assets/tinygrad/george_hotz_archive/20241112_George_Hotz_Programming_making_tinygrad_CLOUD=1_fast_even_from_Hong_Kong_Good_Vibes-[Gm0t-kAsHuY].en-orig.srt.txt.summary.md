# George Hotz Livestream Summary

## Technical Focus
- George worked on improving tinygrad's `CLOUD=1` feature to make it faster, especially when used from remote locations like Hong Kong
- He implemented HTTP pipelining/request batching to reduce round-trip latency between client and server
- The code changes allow operations to be queued and executed in batches instead of making individual HTTP requests for each operation
- He demonstrated training an MNIST model remotely on a server in America and even on his mobile phone

## About tinygrad
- tinygrad is a lightweight ML framework written entirely in Python (~10,000 lines)
- Has no dependencies (not even NumPy is required anymore)
- Features include:
  - Visualization tools for kernel operations
  - Support for various hardware backends
  - A "sovereign stack" that doesn't rely on other frameworks
  - Cross-device compatibility (works on Windows, mobile phones)
  - GPU acceleration

## Business Philosophy
- Plans for a tinygrad cloud service (~50 cents/hour, billed per second)
- Hiring approach based on merit through bounties and contributions
- Skeptical about raising excessive capital: "I have $4.2 million in a bank account I have no idea what to do with"
- Values code quality and simplicity over rapid expansion

## Personal Commentary
- Currently in Hong Kong but plans to return to America
- Shared positive views on Trump's election victory and Elon Musk
- Appreciates Hong Kong's efficiency, density, and cash-based economy
- Criticizes bureaucracy and advocates for smaller government
- Discussed his views on free speech, immigration, and economic policies

## Development Approach
- Emphasized clean, pythonic code over complex abstractions
- Uses extensive visualization and debugging tools
- Maintains MIT licensing for tinygrad
- Team of about five core contributors
- No interest in typical corporate roles (scrum masters, agile coaches)
- Open development process with public meetings

The stream concluded with George discussing future plans for tinygrad's cloud features and the possibility of creating a decentralized GPU cloud service.