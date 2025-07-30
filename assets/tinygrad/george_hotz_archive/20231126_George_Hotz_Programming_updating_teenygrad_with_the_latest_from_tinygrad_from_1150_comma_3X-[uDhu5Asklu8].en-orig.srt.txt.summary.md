# George Hotz Programming Stream Summary

## Main Focus: Updating teenygrad with tinygrad features

George Hotz streamed a programming session focused on updating teenygrad (a simplified version of tinygrad) with the latest features from tinygrad. The work involved:

- Importing tensor, mlops, and optimizer functionality from tinygrad to teenygrad
- Adding type support and fixing various compatibility issues
- Running tests to ensure everything worked properly
- teenygrad grew from 833 to around 937 lines (still maintaining its goal of being under 1000 lines)
- Added support for D-types, casting operations, and fixed various bugs

## Technical Highlights

- Demonstrated training on MNIST dataset with both libraries
- teenygrad is a simplified frontend that can do everything tinygrad can, but isn't optimized for speed
- tinygrad is maintained under 5,000 lines (with a warning light in the office if it goes over)
- Showed Metal's batch executor performance with debugging output
- Added pre-commit hooks and CI integration for teenygrad

## Other Topics Discussed

- Updates on tiny Corp office and internship program (paying $2K/week for interns)
- Commentary on OpenAI leadership changes and AI development philosophy
- Discussion about AI safety, defining real AI risks vs. exaggerated concerns
- Personal philosophy on focusing on what you can change (values and focus) rather than what you can't (intelligence)
- Commentary on money, lifestyle choices, and career decisions
- Promotion of comma.ai devices (mentioned Black Friday sale at $1150)

The stream demonstrated George's hands-on approach to maintaining clean, minimal code while still implementing complex functionality.