# George Hotz Livestream Summary: Edge TPU Reverse Engineering

This livestream shows George Hotz (geohot) working on reverse engineering the Google Coral Edge TPU while discussing his departure from Comma.ai and plans for his new "tiny Corporation."

## Technical Work
- Attempting to get Google's Coral Edge TPU working with TinyGrad (his minimal ML framework)
- Successfully using the Edge TPU to recognize images (banana, apple, chicken) with pre-trained models
- Configuring Docker to run Linux-only tools on his Mac for the Edge TPU compiler
- Examining USB communication between the host and Edge TPU
- Attempting to find hidden debug flags in Google's closed-source compiler
- Analyzing instruction formats sent to the TPU through USB

## TinyGrad Discussion
- Explains how TinyGrad (his ~1000-line ML framework) improves on PyTorch/TensorFlow
- Demonstrates how TinyGrad fuses operations that would normally require multiple GPU kernel launches
- Discusses how TinyGrad's lazy evaluation provides performance advantages
- Mentions potential business opportunities porting TinyGrad to various accelerators

## Personal/Business Updates
- Explains his departure from Comma.ai: "I'm still on the board... I just got bored"
- Clarifies Comma.ai is stable with ~$7M in the bank and 20+ employees
- Discusses launching the "tiny Corporation" focused on ML software for accelerators
- Expresses interest in contracts (not investment) from companies like Nvidia or Google
- Repeatedly mentions wanting to buy a Porsche as part of his "midlife crisis"

The stream demonstrates George's technical approach combining programming, hardware knowledge, and reverse engineering while providing context around his career transitions and personal interests.